# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-row-count bf16 GEMM with cuBLASLt's bits.

``y = x @ w.T`` with one fp32 accumulator per output tile that walks K in
ascending order. Tensor-core GEMMs without split-K give the same bits whatever
their tile shape, so this kernel returns exactly what cuBLASLt returns without
split-K (the deterministic-serving setup) and can replace it at any row count
without changing outputs.

At a few rows cuBLASLt's kernels for some Qwen4Exp shapes run well above the
floor set by the dependent MMA chain over K. This kernel uses 16-row tiles fed
by a deep TMA pipeline, launches dependents at entry (PDL), and prefetches its
weight rows into L2 before waiting on the previous kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@dataclass(frozen=True, slots=True)
class TmaGemmConfig:
    # Largest row count this config serves; larger calls use cuBLASLt.
    max_rows: int
    block_n: int
    block_k: int
    num_stages: int
    l2_prefetch: bool
    block_m: int = 16
    num_warps: int = 4


@triton.jit(do_not_specialize=["M"])
def _tma_gemm_kernel(
    x_desc,
    w_desc,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_w,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    L2_PREFETCH: tl.constexpr,
    PREFETCH_COLS: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    if L2_PREFETCH:
        # One prefetch per 128 bytes of this CTA's weight rows; the weights do
        # not depend on the previous kernel.
        cols = off_n + tl.arange(0, BLOCK_N)
        lines = tl.arange(0, PREFETCH_COLS) * 64
        pf = w_ptr + cols[:, None].to(tl.int64) * stride_w + lines[None, :]
        pf = tl.where((cols[:, None] < N) & (lines[None, :] < K), pf, w_ptr)
        tl.inline_asm_elementwise(
            "prefetch.global.L2 [$1]; mov.u32 $0, 0;",
            "=r,l",
            [pf],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = x_desc.load([off_m, kt * BLOCK_K])
        b = w_desc.load([off_n, kt * BLOCK_K])
        acc = tl.dot(a, b.T, acc)
    rows = off_m + tl.arange(0, BLOCK_M)
    out_cols = off_n + tl.arange(0, BLOCK_N)
    tl.store(
        y_ptr + rows[:, None].to(tl.int64) * stride_y + out_cols[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (out_cols[None, :] < N),
    )


def tma_gemm_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """TMA needs 16-byte aligned bases and row strides and unit column strides."""
    return (
        x.dim() == 2
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.stride(1) == 1
        and weight.stride(1) == 1
        and (x.stride(0) * x.element_size()) % 16 == 0
        and (weight.stride(0) * weight.element_size()) % 16 == 0
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def tma_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    config: TmaGemmConfig,
    launch_pdl: bool,
) -> torch.Tensor:
    from triton.tools.tensor_descriptor import TensorDescriptor

    m, k = x.shape
    n = weight.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    x_desc = TensorDescriptor.from_tensor(x, [config.block_m, config.block_k])
    w_desc = TensorDescriptor.from_tensor(weight, [config.block_n, config.block_k])
    grid = (triton.cdiv(n, config.block_n), triton.cdiv(m, config.block_m))
    _tma_gemm_kernel[grid](
        x_desc,
        w_desc,
        weight,
        y,
        m,
        n,
        k,
        weight.stride(0),
        y.stride(0),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        NUM_STAGES=config.num_stages,
        L2_PREFETCH=config.l2_prefetch,
        PREFETCH_COLS=triton.next_power_of_2(triton.cdiv(k, 64)),
        launch_pdl=launch_pdl,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return y
