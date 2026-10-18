# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The one-kernel shared-expert sigmoid gate against the native GEMM, sigmoid
and multiply: identical up to one bf16 ulp of the logit, at decode and
prefill token counts."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.shared_expert_gate import (
    can_fuse_shared_gate,
    shared_gate_mul,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")


@pytest.mark.parametrize("tokens", [1, 8, 15, 64, 75, 300, 2048])
@pytest.mark.parametrize("hidden", [2560, 4096])
def test_shared_gate_mul_matches_native(tokens: int, hidden: int) -> None:
    torch.manual_seed(tokens)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    out = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    assert can_fuse_shared_gate(x, w, out)

    native = F.sigmoid(F.linear(x, w)) * out
    fused = shared_gate_mul(x, w, out)

    # Only the dot product's summation order differs: at most one bf16 ulp of
    # the logit, which moves the gate (and so the product) by about as much.
    torch.testing.assert_close(fused, native, rtol=2**-7, atol=1e-3)
    assert (fused == native).float().mean() > 0.95


def test_shared_gate_mul_handles_strided_rows_and_empty_batches() -> None:
    hidden = 2560
    x = torch.randn(16, hidden * 2, device="cuda", dtype=torch.bfloat16)[:, :hidden]
    w = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    out = torch.randn(16, hidden, device="cuda", dtype=torch.bfloat16)
    assert can_fuse_shared_gate(x, w, out)
    torch.testing.assert_close(
        shared_gate_mul(x, w, out),
        F.sigmoid(F.linear(x, w)) * out,
        rtol=2**-7,
        atol=1e-3,
    )
    empty = torch.empty(0, hidden, device="cuda", dtype=torch.bfloat16)
    assert shared_gate_mul(empty, w, empty).shape == (0, hidden)


def test_can_fuse_rejects_other_layouts() -> None:
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    out = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    assert not can_fuse_shared_gate(x.float(), torch.ones(1, 64, device="cuda"), out)
    w2 = torch.ones(2, 64, device="cuda", dtype=torch.bfloat16)
    assert not can_fuse_shared_gate(x, w2, out)
    assert not can_fuse_shared_gate(x.t(), torch.ones(1, 4, device="cuda"), out)
