# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-kernel sigmoid gate for Qwen2-MoE-style shared experts.

The shared expert's output is scaled by ``sigmoid(x @ expert_gate.T)``, a
one-column projection of the MoE input. Natively that is a GEMM with N = 1
(a split-K GEMM plus its reduce kernel at decode sizes), a sigmoid over the
[tokens, 1] logits and a broadcast multiply. This kernel computes each
token's gate dot product, its sigmoid and the scaled row in one launch, with
the native path's bf16 roundings (the logit, the gate and the product), so it
differs only where the dot product's summation order moves the logit by one
bf16 ulp.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _shared_gate_mul_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_x,
    stride_out,
    stride_y,
    K,
    H,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + offs_k
        mask = k < K
        xv = tl.load(x_ptr + row.to(tl.int64) * stride_x + k, mask=mask, other=0.0)
        wv = tl.load(w_ptr + k, mask=mask, other=0.0)
        acc += xv.to(tl.float32) * wv.to(tl.float32)
    logit = tl.sum(acc, axis=0).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(logit).to(tl.bfloat16).to(tl.float32)
    cols = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    cmask = cols < H
    v = tl.load(out_ptr + row.to(tl.int64) * stride_out + cols, mask=cmask, other=0.0)
    y = (gate * v.to(tl.float32)).to(tl.bfloat16)
    tl.store(y_ptr + row.to(tl.int64) * stride_y + cols, y, mask=cmask)


def can_fuse_shared_gate(
    x: torch.Tensor, gate_weight: torch.Tensor, out: torch.Tensor
) -> bool:
    """Whether shared_gate_mul takes these: bf16 2-D rows with unit last
    stride and a [1, K] bf16 gate weight on the same CUDA device."""
    return (
        x.is_cuda
        and x.dim() == 2
        and out.dim() == 2
        and x.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and gate_weight.dtype == torch.bfloat16
        and gate_weight.shape == (1, x.shape[1])
        and gate_weight.is_contiguous()
        and x.stride(1) == 1
        and out.stride(1) == 1
        and out.shape[0] == x.shape[0]
    )


def shared_gate_mul(
    x: torch.Tensor, gate_weight: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """sigmoid(x @ gate_weight.T) * out for x [T, K], gate_weight [1, K] and
    out [T, H], all bf16; returns a new [T, H] tensor."""
    tokens, k = x.shape
    h = out.shape[1]
    y = torch.empty_like(out)
    if tokens == 0:
        return y
    # Few tokens: split each row across column blocks so more programs stream
    # it (each recomputes the dot product from L2); many: one per row.
    block_h = 512 if tokens <= 64 else triton.next_power_of_2(h)
    grid = (tokens, triton.cdiv(h, block_h))
    _shared_gate_mul_kernel[grid](
        x,
        gate_weight,
        out,
        y,
        x.stride(0),
        out.stride(0),
        y.stride(0),
        k,
        h,
        BLOCK_K=triton.next_power_of_2(min(k, 4096)),
        BLOCK_H=block_h,
        num_warps=4,
    )
    return y
