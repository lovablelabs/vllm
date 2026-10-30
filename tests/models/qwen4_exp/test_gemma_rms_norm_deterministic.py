# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one-kernel Gemma RMSNorm of deterministic serving gives a row the same
bits at any batch size and matches the native norm up to rounding."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.gemma_rms_norm import gemma_rms_norm
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("needs CUDA", allow_module_level=True)


def _native(x, weight, eps, group):
    xf = x.float().unflatten(-1, (-1, group))
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = (xf * torch.rsqrt(variance + eps)).flatten(-2)
    return (xf * (weight.float() + 1.0)).to(x.dtype)


@pytest.mark.parametrize("hidden,group", [(2560, 2560), (10240, 10240), (10240, 2560)])
def test_gemma_rms_norm_rows_are_batch_invariant(hidden, group):
    gen = torch.Generator(device="cuda").manual_seed(hidden + group)
    x = torch.randn(300, hidden, device="cuda", generator=gen).bfloat16() * 3
    weight = (torch.randn(hidden, device="cuda", generator=gen) * 0.1).bfloat16()
    full = gemma_rms_norm(x, weight, 1e-6, group)
    for rows in (1, 2, 7, 64):
        assert torch.equal(gemma_rms_norm(x[:rows], weight, 1e-6, group), full[:rows])
    # Strided rows (a view into a wider buffer) give the same bits.
    wide = torch.zeros(300, hidden + 64, device="cuda", dtype=x.dtype)
    wide[:, :hidden] = x
    assert torch.equal(gemma_rms_norm(wide[:, :hidden], weight, 1e-6, group), full)
    torch.testing.assert_close(full, _native(x, weight, 1e-6, group), atol=2e-2, rtol=1e-2)
