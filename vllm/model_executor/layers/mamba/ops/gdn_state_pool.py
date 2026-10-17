# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN chunked prefill on FlashInfer's SM100 state-pool mode.

The gather path collects the prefill rows' states from the SSM cache, zeroes the rows
of new requests (a boolean-mask assignment, which syncs with the host), casts them to
fp32, runs the kernel, casts the final states back and scatters them into the cache,
all per layer, and casts cu_seqlens to int64 (which the SM100 kernel casts back to
int32). FlashInfer's pool mode reads and writes the cache rows in place
(state_indices, output_state = the pool) and is bit-identical on B200. Only the new
rows' zeroing remains, done here on the device.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _zero_new_state_rows_kernel(
    pool_ptr,
    idx_ptr,
    has_init_ptr,
    row_stride,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    if tl.load(has_init_ptr + seq):
        return
    row = tl.load(idx_ptr + seq).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    zeros = tl.zeros([BLOCK], dtype=pool_ptr.dtype.element_ty)
    tl.store(pool_ptr + row * row_stride + offs, zeros, mask=offs < ROW)


def zero_new_state_rows(
    pool: torch.Tensor, state_indices: torch.Tensor, has_initial_state: torch.Tensor
) -> None:
    """pool[state_indices[i]] = 0 where not has_initial_state[i]; no host sync."""
    n = state_indices.shape[0]
    if n == 0:
        return
    row = pool[0].numel()
    block = 4096
    _zero_new_state_rows_kernel[(n, triton.cdiv(row, block))](
        pool,
        state_indices,
        has_initial_state,
        pool.stride(0),
        ROW=row,
        BLOCK=block,
        num_warps=4,
    )


def pool_mode_supported(pool: torch.Tensor) -> bool:
    """FlashInfer's pool mode wants each row's [H, V, K] block contiguous."""
    return (
        pool.dim() == 4
        and pool[0].is_contiguous()
        and pool.dtype
        in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        )
    )


def fi_chunk_gated_delta_rule_pool(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    pool: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
) -> torch.Tensor:
    """FlashInfer GDN prefill reading and writing the prefill rows' states in the
    SSM cache in place; returns the output (1, T, H, V). Inputs are prepared
    exactly as fi_chunk_gated_delta_rule prepares them."""
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()
    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    # The zeroing kernel and FlashInfer's pool mode both index these densely; the
    # metadata may hand over strided views.
    state_indices = state_indices.to(torch.int32).contiguous()
    has_initial_state = has_initial_state.contiguous()
    zero_new_state_rows(pool, state_indices, has_initial_state)
    output, _ = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(g.to(torch.float32)),
        beta=beta.to(torch.float32),
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens
        if cu_seqlens.dtype == torch.int32
        else cu_seqlens.to(torch.int32),
        output_state=pool,
        state_indices=state_indices,
        # "auto" may pick a backend that copies the sequence lengths to the
        # host on every call; the SM100 kernel takes them on the device.
        backend="flashinfer",
    )
    return output.unsqueeze(0)
