# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-kernel Gemma RMSNorm for the eager Qwen4Exp NVIDIA model.

``y = (x / rms(x) * (1 + w)).to(x.dtype)`` over groups of ``group_size``
columns, computed in fp32 with the weight's ``1 + w`` formed in the kernel.
One program per (row, group) with a launch configuration fixed by the group
width, so a row's result does not depend on the batch around it.

The NVIDIA model runs eagerly, so the PyTorch-native norm (float cast,
square, mean, add, rsqrt, two multiplies, cast and the ``1 + w`` weight)
is eight to ten kernels per call; with deterministic serving its mean is
the batch-invariant Triton kernel. This kernel replaces all of them in
deterministic serving, where the summation order is free to change as long
as it does not depend on the batch.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gemma_rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    eps,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < GROUP
    offs = group * GROUP + cols
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / GROUP
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
    y = x * tl.math.rsqrt(variance + eps) * w
    tl.store(y_ptr + row * stride_y + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def gemma_rms_norm_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.stride(1) == 1
        and weight.is_contiguous()
        and weight.numel() == x.shape[1]
    )


def gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int | None = None,
) -> torch.Tensor:
    """Gemma RMSNorm of x [T, H] with weight [H] over groups of group_size
    columns (the whole row by default)."""
    tokens, hidden = x.shape
    group = hidden if group_size is None else group_size
    assert hidden % group == 0
    y = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if tokens == 0:
        return y
    block = triton.next_power_of_2(group)
    _gemma_rms_norm_kernel[(tokens, hidden // group)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        eps,
        GROUP=group,
        BLOCK=block,
        num_warps=8 if block >= 8192 else 4,
    )
    return y
