# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA HyperConnection kernels for Qwen4Exp."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(GROUP_DIM)

    pid = tl.program_id(0)
    group_id = pid % NUM_GROUPS
    row = pid // NUM_GROUPS

    offs_g = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * GROUP_DIM + offs_g
    mask = offs_g < GROUP_DIM
    # A [GROUP_DIM] affine is shared; a [DIM] affine follows the grouped
    # checkpoint layout.
    w_offs = offs_g if W_SHARED else offsets

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + w_offs, mask, other=0.0)

    rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    # Gemma's (1 + w) affine is written this way to lower to an FMA.
    y = x * rrms
    y += y * w.to(tl.float32)

    tl.store(y_ptr + row * stride_y + offsets, y, mask)


def _grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    N, DIM = x.shape
    assert x.stride(1) == 1, "grouped Gemma RMSNorm requires unit inner stride"
    assert weight.is_contiguous(), "grouped Gemma RMSNorm weight must be contiguous"
    assert DIM % num_groups == 0
    group_dim = DIM // num_groups
    assert weight.numel() in (group_dim, DIM)

    y = x.new_empty(x.shape)
    _grouped_gemma_rmsnorm_kernel[(N * num_groups,)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        DIM,
        num_groups,
        W_SHARED=weight.numel() == group_dim,
        EPS=eps,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return y


@triton.jit
def _hc_silu_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(DIM)

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + row * stride_x + offs, mask).to(tl.float32) / HC
    y = x * tl.sigmoid(x)

    tl.store(y_ptr + row * stride_y + offs, y, mask)


def _hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    num_tokens, DIM = x.shape
    assert x.stride(1) == 1

    output = x.new_empty(x.shape)
    _hc_silu_kernel[(num_tokens,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        DIM=DIM,
        HC=hc_count,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return output


# Down projections past the fused CuTe-DSL range (cute_dsl/hc_down_silu.py)
# and up to HC_DOWN_SLICED_MAX_TOKENS run as one batched GEMM over K slices
# with fp32 partials, which _hc_down_silu_sliced_kernel sums and activates.
# cuBLAS would otherwise pick a split-K GEMM and its reduce kernel at these
# sizes, and hc_silu adds a third launch.
HC_DOWN_SLICES = 8
HC_DOWN_SLICED_MAX_TOKENS = 512


@triton.jit
def _hc_down_silu_sliced_kernel(
    part_ptr,
    y_ptr,
    stride_part_slice,
    stride_part_row,
    stride_y,
    SLICES: tl.constexpr,
    RANK: tl.constexpr,
    WIDTH: tl.constexpr,
    HC: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(WIDTH)

    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < WIDTH

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for s in tl.static_range(SLICES):
        acc += tl.load(
            part_ptr + s * stride_part_slice + row * stride_part_row + offs,
            mask=mask,
            other=0.0,
        )
    # Round as the unsliced GEMM's bf16 output does, then apply hc_silu to the
    # low-rank columns; the injection logits stay plain.
    down = acc.to(y_ptr.dtype.element_ty).to(tl.float32)
    x = down / HC
    y = tl.where(offs < RANK, x * tl.sigmoid(x), down)

    tl.store(y_ptr + row * stride_y + offs, y, mask)


def _hc_down_silu_sliced(
    x: torch.Tensor, weight: torch.Tensor, rank: int, hc_count: int
) -> torch.Tensor:
    """Return ``x @ weight.T`` for the merged down+inject weight [N, K].

    The output is [M, N] bf16 with hc_silu applied to the first ``rank``
    columns; the padding columns past rank + hc_count are left unwritten. The
    padded row stride keeps the low-rank view aligned for the up projection's
    GEMM (a 324-column stride sends cuBLAS to a slower sm80 kernel).
    """
    num_tokens, k = x.shape
    n = weight.shape[0]
    assert x.stride(1) == 1 and weight.is_contiguous()
    assert k % HC_DOWN_SLICES == 0 and rank + hc_count <= n

    from .cute_dsl.hc_down_silu import (
        hc_down_silu_sliced_mma,
        hc_down_silu_sliced_mma_ok,
    )

    # Few rows: the same slices, partial order and epilogue in one clustered
    # launch (same bits, so the choice may follow the row count).
    if hc_down_silu_sliced_mma_ok(x, weight, HC_DOWN_SLICES):
        return hc_down_silu_sliced_mma(x, weight, rank, hc_count, HC_DOWN_SLICES)

    slice_k = k // HC_DOWN_SLICES
    x_slices = x.unflatten(1, (HC_DOWN_SLICES, slice_k)).transpose(0, 1)
    w_slices = weight.view(n, HC_DOWN_SLICES, slice_k).permute(1, 2, 0)
    partials = torch.bmm(x_slices, w_slices, out_dtype=torch.float32)

    output = x.new_empty((num_tokens, n))
    _hc_down_silu_sliced_kernel[(num_tokens,)](
        partials,
        output,
        partials.stride(0),
        partials.stride(1),
        output.stride(0),
        SLICES=HC_DOWN_SLICES,
        RANK=rank,
        WIDTH=rank + hc_count,
        HC=hc_count,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return output


@triton.jit
def _hc_gate_mix_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x,
    stride_g,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_DIM: tl.constexpr = DIM // HC

    row = tl.program_id(0)
    tile_id = tl.program_id(1)
    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_inner < HC_DIM

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    # The constexpr loop is unrolled and keeps one stream live at a time.
    # Materializing [HC, BLOCK_SIZE] more than doubles latency at large M.
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * HC_DIM + offs_inner
        g = tl.load(g_ptr + row * stride_g + offsets, mask, other=0.0)
        x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc /= HC

    tl.store(y_ptr + row * stride_y + offs_inner, acc, mask)


def _hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    N, DIM = gate.shape
    assert x.shape == gate.shape
    assert DIM % hc_count == 0
    assert x.stride(1) == 1
    assert gate.stride(1) == 1

    HC_DIM = DIM // hc_count
    out = x.new_empty(N, HC_DIM)
    BLOCK_SIZE = 512
    _hc_gate_mix_kernel[(N, triton.cdiv(HC_DIM, BLOCK_SIZE))](
        x,
        gate,
        out,
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        DIM,
        hc_count,
        BLOCK_SIZE,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out


@triton.jit
def _hc_combine_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    out_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)

    row = tl.program_id(0)
    tile_id = tl.program_id(1)

    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_inner = offs_inner < HC_DIM
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    offs = offs_hc[:, None] * HC_DIM + offs_inner[None, :]
    mask = mask_hc[:, None] & mask_inner[None, :]

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    if inj_ptr is not None:
        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    res = tl.load(res_ptr + row * stride_res + offs, mask, other=0.0)

    # Keeping HC as a broadcast dimension is faster here than four separate
    # residual load/store sequences.
    if inj_ptr is not None:
        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
        block = block.to(tl.float32)[None, :] * inj[:, None]
    out = res.to(tl.float32) + block.to(tl.float32)

    tl.store(out_ptr + row * stride_out + offs, out, mask=mask)


def _hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    if injection_logits is not None:
        assert injection_logits.shape == (N, hc_count)
        assert injection_logits.stride(1) == 1

    stride_injection = injection_logits.stride(0) if injection_logits is not None else 0

    out = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    _hc_combine_kernel[(N, triton.cdiv(hc_dim, BLOCK_SIZE))](
        block_output,
        residual,
        injection_logits,
        out,
        block_output.stride(0),
        residual.stride(0),
        stride_injection,
        out.stride(0),
        hc_dim,
        hc_count,
        BLOCK_SIZE,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out


@triton.jit
def _hc_combine_norm_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

    pid = tl.program_id(0)
    row = pid // HC
    stream = pid % HC
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    # Shared norm weights repeat across streams; per-branch weights use the
    # same flattened HC layout as the residual.
    w_offs = offs_inner if W_SHARED else offs

    if launch_pdl:
        # The next kernel (the HC down) may start now and fetch its weights;
        # its griddepcontrol.wait still waits for this grid.
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()

    # Start the uncached residual load first, then issue the other combine
    # loads before consuming any of them.
    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    if inj_ptr is not None:
        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(
        block_ptr + row * stride_block + offs_inner,
        mask_inner,
        other=0.0,
    )
    if inj_ptr is not None:
        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
        block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    # Round the materialized combine result before normalization. This matches
    # the unfused combine -> RMSNorm boundary.
    out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)
    if inj_ptr is None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

    out = out.to(tl.float32)
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    if inj_ptr is not None:
        # Loading the weight earlier helps decode but keeps the tile live across
        # the reduction and regresses larger batches, so defer it to the norm.
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offs, y, mask_inner)


def _hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    if injection_logits is not None:
        assert injection_logits.shape == (N, hc_count)
        assert injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous()
    assert norm_weight.numel() in (hc_dim, DIM)

    stride_injection = injection_logits.stride(0) if injection_logits is not None else 0

    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    _hc_combine_norm_kernel[(N * hc_count,)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        block_output.stride(0),
        residual.stride(0),
        stride_injection,
        out.stride(0),
        y.stride(0),
        hc_dim,
        hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out, y


def _same_shape_fake(x: torch.Tensor, *args) -> torch.Tensor:
    return x.new_empty(x.shape)


def _hc_down_silu_sliced_fake(
    x: torch.Tensor, weight: torch.Tensor, rank: int, hc_count: int
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def _hc_gate_mix_fake(
    x: torch.Tensor, gate: torch.Tensor, hc_count: int
) -> torch.Tensor:
    del gate
    return x.new_empty((x.shape[0], x.shape[1] // hc_count))


def _hc_combine_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    del block_output, injection_logits, hc_count
    return residual.new_empty(residual.shape)


def _hc_combine_norm_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    return residual.new_empty(residual.shape), residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="qwen4_exp_grouped_gemma_rmsnorm",
    op_func=_grouped_gemma_rmsnorm,
    fake_impl=_same_shape_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_silu",
    op_func=_hc_silu,
    fake_impl=_same_shape_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_down_silu_sliced",
    op_func=_hc_down_silu_sliced,
    fake_impl=_hc_down_silu_sliced_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_gate_mix",
    op_func=_hc_gate_mix,
    fake_impl=_hc_gate_mix_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine",
    op_func=_hc_combine,
    fake_impl=_hc_combine_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine_norm",
    op_func=_hc_combine_norm,
    fake_impl=_hc_combine_norm_fake,
)


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_grouped_gemma_rmsnorm(x, weight, eps, num_groups)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_silu(x, hc_count)


def hc_down_silu_sliced(
    x: torch.Tensor, weight: torch.Tensor, rank: int, hc_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The HC down projection's activated low-rank columns [M, rank] and its
    injection logits [M, hc_count], from the K-sliced GEMM."""
    output = torch.ops.vllm.qwen4_exp_hc_down_silu_sliced(x, weight, rank, hc_count)
    return output[:, :rank], output[:, rank : rank + hc_count]


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_gate_mix(x, gate, hc_count)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    return torch.ops.vllm.qwen4_exp_hc_combine(
        residual, block_output, injection_logits, hc_count
    )


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.qwen4_exp_hc_combine_norm(
        residual,
        block_output,
        injection_logits,
        norm_weight,
        eps,
        hc_count,
    )


__all__ = [
    "HC_DOWN_SLICED_MAX_TOKENS",
    "HC_DOWN_SLICES",
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_down_silu_sliced",
    "hc_gate_mix",
    "hc_silu",
]
