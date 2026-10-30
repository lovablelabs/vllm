# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen4Exp MTP drafter's FP8 head: under deterministic serving a row's
logits keep their bits at any row count, and they pick the bf16 head's top
token."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda() or not current_platform.has_device_capability(89):
    pytest.skip("FP8 GEMMs need CUDA sm89+", allow_module_level=True)


def test_fp8_draft_head_rows_are_batch_invariant(monkeypatch):
    import vllm.envs as envs
    from vllm.models.qwen4_exp.nvidia.mtp import _Fp8DraftHead

    monkeypatch.setattr(envs, "VLLM_DETERMINISTIC_SERVING", True)

    gen = torch.Generator(device="cuda").manual_seed(0)
    vocab, hidden = 16384, 2560
    weight = (torch.randn(vocab, hidden, device="cuda", generator=gen) * 0.02).bfloat16()
    head = _Fp8DraftHead(SimpleNamespace(weight=weight, tp_size=1, shard_indices=None))
    # Hidden states near embedding rows, so each row has a clear top token.
    rows = torch.randint(0, vocab, (256,), device="cuda", generator=gen)
    x = (weight[rows].float() * 30 + torch.randn(256, hidden, device="cuda", generator=gen) * 0.3).bfloat16()
    full = head.apply(None, x)
    for m in (1, 2, 4, 16, 32, 48, 49, 64, 128, 200):
        assert torch.equal(head.apply(None, x[:m]), full[:m]), m
    bf16_top = (x @ weight.t()).argmax(dim=-1)
    assert (full.argmax(dim=-1) == bf16_top).float().mean().item() > 0.99
