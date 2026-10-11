# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auxiliary extraction must preserve the target's delayed residual path."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import model as qwen4


class _Mixer:
    @staticmethod
    def combine(hidden, block, injection):
        return hidden if block is None else hidden + block * injection

    def combine_and_mix(self, hidden, block, injection):
        materialized = self.combine(hidden, block, injection)
        return materialized, materialized.view(-1, 2, 3).mean(1), None


class _Layer(nn.Module):
    ple = None
    mlp_hyper_connection = _Mixer()

    def forward(self, hidden_states, prev_block_output, prev_injection, **kwargs):
        hidden = self.mlp_hyper_connection.combine(
            hidden_states, prev_block_output, prev_injection
        )
        block = hidden * 0.25 + 1
        injection = torch.tensor([1, 2, 3, 4, 5, 6], dtype=hidden.dtype)
        return hidden, block, injection


@pytest.mark.parametrize("deepstack", [False, True])
def test_aux_capture_preserves_target_and_mtp_state(monkeypatch, deepstack):
    """Capture completed states, including external additions, without feedback."""
    monkeypatch.setattr(
        qwen4,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    model = qwen4.Qwen4ExpModel.__new__(qwen4.Qwen4ExpModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hc_count=2, hidden_size=3)
    model.layers = nn.ModuleList([_Layer(), _Layer()])
    model.start_layer, model.end_layer = 0, 2
    model.hyper_connection_mixer = _Mixer()
    model._mtp_hidden_buffer = torch.empty(2, 6)
    embeds = torch.arange(6, dtype=torch.float32).view(2, 3)
    extra = {"deepstack_input_embeds_0": embeds + 0.5} if deepstack else None
    kwargs = dict(
        input_ids=None,
        positions=torch.arange(2),
        inputs_embeds=embeds,
        deepstack_input_embeds=extra,
    )
    model.aux_hidden_state_layers = ()
    baseline = model(**kwargs)
    baseline_mtp = model._mtp_hidden_buffer.clone()
    model.aux_hidden_state_layers = (0, 1, 2)
    output, features = model(**kwargs)
    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    torch.testing.assert_close(model._mtp_hidden_buffer, baseline_mtp, rtol=0, atol=0)
    expected = embeds.repeat(1, 2)
    torch.testing.assert_close(features[0], embeds, rtol=0, atol=0)
    for index in range(2):
        expected = expected + (expected * 0.25 + 1) * torch.arange(1, 7)
        if index == 0 and extra is not None:
            expected = expected + extra["deepstack_input_embeds_0"].repeat(1, 2)
        torch.testing.assert_close(
            features[index + 1], expected.view(2, 2, 3).mean(1), rtol=0, atol=0
        )
