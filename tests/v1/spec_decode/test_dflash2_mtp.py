# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 heads that draft through the target's MTP layer."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2_mtp import (
    DFlash2MTPBridge,
    get_pos1_source,
    split_mtp_overrides,
)
from vllm.models.qwen4_exp.nvidia.model import _drafter_runs_mtp_layer
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash2.mtp_speculator import (
    DFlash2MTPSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


def _entry(tensor: torch.Tensor, checkpoint_tensor: str) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "checkpoint_tensor": checkpoint_tensor,
    }


def test_split_mtp_overrides_maps_and_splits_checkpoint_sources():
    norm = torch.randn(8, dtype=torch.bfloat16)
    gate_up = torch.randn(6, 4, dtype=torch.bfloat16)
    gate_ref = (
        "cat([mtp.layers.0.mlp.shared_expert.gate_proj.weight, "
        "mtp.layers.0.mlp.shared_expert.up_proj.weight], 0) [3, 4] [3, 4]"
    )
    table = {
        "mtp_head.mtp.pre_fc_norm_hidden": _entry(
            norm, "mtp.pre_fc_norm_hidden.weight"
        ),
        "mtp_head.tail.shared_gate_up": _entry(gate_up, gate_ref),
    }
    overrides = {
        "mtp_head.mtp.pre_fc_norm_hidden": norm,
        "mtp_head.tail.shared_gate_up": gate_up,
    }

    pairs = dict(split_mtp_overrides(overrides, table))

    assert pairs.keys() == {
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    }
    assert torch.equal(pairs["mtp.pre_fc_norm_hidden.weight"], norm)
    assert torch.equal(
        pairs["mtp.layers.0.mlp.shared_expert.gate_proj.weight"], gate_up[:3]
    )
    assert torch.equal(
        pairs["mtp.layers.0.mlp.shared_expert.up_proj.weight"], gate_up[3:]
    )


@pytest.mark.parametrize("mismatch", ["missing", "unknown", "shape", "dtype"])
def test_split_mtp_overrides_rejects_a_table_mismatch(mismatch: str):
    tensor = torch.zeros(4, dtype=torch.bfloat16)
    table = {"mtp_head.mtp.q_norm": _entry(tensor, "mtp.layers.0.q_norm.weight")}
    overrides = {"mtp_head.mtp.q_norm": tensor}
    if mismatch == "missing":
        overrides = {}
    elif mismatch == "unknown":
        overrides["mtp_head.mtp.k_norm"] = tensor
    elif mismatch == "shape":
        overrides["mtp_head.mtp.q_norm"] = torch.zeros(5, dtype=torch.bfloat16)
    else:
        overrides["mtp_head.mtp.q_norm"] = tensor.float()

    with pytest.raises(ValueError):
        split_mtp_overrides(overrides, table)


def test_pos1_source_follows_the_head_config():
    assert get_pos1_source({}) is None
    assert get_pos1_source({"pos1_mtp_pass": False, "pos1": {}}) is None
    assert get_pos1_source({"pos1_mtp_pass": True}) == "mtp"
    assert (
        get_pos1_source({"pos1_mtp_pass": True, "pos1": {"pos1_source": "dense"}})
        == "dense"
    )
    with pytest.raises(ValueError):
        get_pos1_source({"pos1_mtp_pass": True, "pos1": {"pos1_source": "x"}})


@pytest.mark.parametrize(
    ("method", "architectures", "expected"),
    [
        ("mtp", ["Qwen4ExpMTP"], True),
        ("dflash", ["DFlash2MTPDraftModel"], True),
        ("dflash", ["DFlash2DraftModel"], False),
        ("eagle3", ["DFlash2MTPDraftModel"], False),
    ],
)
def test_qwen4_exp_keeps_multi_stream_states_for_mtp_drafters(
    method: str, architectures: list[str], expected: bool
):
    spec_config = SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(architectures=architectures),
    )
    assert _drafter_runs_mtp_layer(spec_config) is expected


def _speculator(pos1_source: str | None, num_reqs: int, top_k: int, hidden: int):
    """A DFlash2MTPSpeculator with only the state the candidate hooks read."""
    speculator = DFlash2MTPSpeculator.__new__(DFlash2MTPSpeculator)
    speculator.pos1_source = pos1_source
    root_ids = torch.arange(num_reqs * top_k).view(num_reqs, top_k) + 1000
    root_logits = torch.randn(num_reqs, top_k)
    speculator.mtp = SimpleNamespace(
        pos1_mixed=torch.randn(num_reqs, hidden),
        root_candidates=lambda n: (root_ids[:n], root_logits[:n]),
    )
    return speculator, root_ids, root_logits


@pytest.mark.parametrize("pos1_source", [None, "dense", "mtp"])
def test_mtp_layer_drafts_the_first_positions(monkeypatch, pos1_source):
    num_reqs, num_steps, top_k, hidden = 3, 4, 5, 8
    speculator, root_ids, root_logits = _speculator(
        pos1_source, num_reqs, top_k, hidden
    )
    seen_rows = []

    def compute_candidates(self, hidden_states):
        seen_rows.append(hidden_states)
        shape = (hidden_states.shape[0], num_steps, top_k)
        return torch.zeros(shape, dtype=torch.int64), torch.zeros(shape)

    monkeypatch.setattr(DFlash2Speculator, "_compute_candidates", compute_candidates)
    hidden_states = torch.randn(num_reqs, num_steps, hidden)

    candidate_ids, unary_logits = speculator._compute_candidates(hidden_states)

    # The anchor row's MTP top-k replaces the dense candidates of step 0.
    assert torch.equal(candidate_ids[:, 0], root_ids)
    assert torch.equal(unary_logits[:, 0], root_logits)
    assert not candidate_ids[:, 1:].any()
    # With pos1_source "mtp" the position-1 row is scored in the same GEMM.
    expected_rows = hidden_states.clone()
    if pos1_source == "mtp":
        expected_rows[:, 1] = speculator.mtp.pos1_mixed
    assert torch.equal(seen_rows[0], expected_rows)


@pytest.mark.parametrize("pos1_source", [None, "mtp"])
def test_mtp_drafted_steps_keep_the_mtp_greedy_choice(monkeypatch, pos1_source):
    num_reqs, num_steps, top_k = 2, 3, 4
    speculator, _, _ = _speculator(pos1_source, num_reqs, top_k, hidden=8)
    selector_scores = torch.randn(num_reqs, num_steps, top_k, top_k)
    monkeypatch.setattr(
        DFlash2Speculator,
        "_score_candidates",
        lambda self, *args: selector_scores.clone(),
    )
    unary_logits = torch.randn(num_reqs, num_steps, top_k)

    scores = speculator._score_candidates(None, unary_logits, None)

    assert torch.equal(scores[:, 0, 0], unary_logits[:, 0])
    forced_steps = 2 if pos1_source == "mtp" else 1
    if pos1_source == "mtp":
        assert torch.equal(scores[:, 1, 0], unary_logits[:, 1])
    assert torch.equal(scores[:, forced_steps:], selector_scores[:, forced_steps:])
    assert torch.equal(
        scores[:, :forced_steps, 1:], selector_scores[:, :forced_steps, 1:]
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_bridge_matches_reference(dist_init):
    torch.manual_seed(0)
    hidden, eps = 64, 1e-6
    # Model loading builds every parameter in the model dtype.
    bridge = DFlash2MTPBridge(hidden, 4, eps, torch.bfloat16).cuda().to(torch.bfloat16)
    bridge.bridge.weight.data.normal_(std=0.05)
    bridge.bridge_norm.weight.data.normal_(mean=1.0, std=0.1)
    base = torch.randn(7, hidden, dtype=torch.bfloat16, device="cuda")
    mixed = torch.randn(7, hidden, dtype=torch.bfloat16, device="cuda")

    out = bridge(base, mixed)

    m = mixed.float()
    normed = (m * torch.rsqrt(m.pow(2).mean(-1, keepdim=True) + eps)).to(
        torch.bfloat16
    ) * bridge.bridge_norm.weight
    delta = (normed @ bridge.bridge.weight.t()).float()
    rms = base.float().pow(2).mean(-1, keepdim=True).sqrt()
    expected = (base.float() + delta * rms).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
