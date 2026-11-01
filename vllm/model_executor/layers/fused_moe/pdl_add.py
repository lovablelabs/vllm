# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared + routed expert output add with programmatic dependent launch.

At decode sizes the add is a ~1.5 us kernel between the routed experts' last
kernel and the all-reduce. torch's elementwise kernel launches without PDL,
so it starts ~0.35 us after its predecessor ends; this one launches with PDL,
lets its dependents launch at its entry and waits before its first load. The
bits are torch's: an fp32 sum rounded once to the output dtype.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_BLOCK = 1024


@triton.jit(do_not_specialize=["n"])
def _add_kernel(
    a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr, launch_pdl: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    mask = offs < n
    a = tl.load(a_ptr + offs, mask).to(tl.float32)
    b = tl.load(b_ptr + offs, mask).to(tl.float32)
    tl.store(out_ptr + offs, a + b, mask)


def add_shared_and_routed(shared: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
    if not (
        shared.is_cuda
        and shared.shape == routed.shape
        and shared.dtype == routed.dtype
        and shared.dtype in (torch.bfloat16, torch.float16)
        and shared.is_contiguous()
        and routed.is_contiguous()
        and current_platform.is_arch_support_pdl()
    ):
        return shared + routed
    out = torch.empty_like(shared)
    n = shared.numel()
    if n:
        _add_kernel[(triton.cdiv(n, _BLOCK),)](
            shared, routed, out, n, _BLOCK, launch_pdl=True, num_warps=4
        )
    return out
