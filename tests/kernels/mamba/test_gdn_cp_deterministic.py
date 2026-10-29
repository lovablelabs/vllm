# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN prefill on FlashInfer's chunk-parallel kernels in one-block segments
(deterministic serving): a sequence's outputs do not depend on the other
sequences in the call, and a prompt prefilled in one call equals the same
prompt prefilled block by block with the state stored to a bf16 cache in
between, or resumed from a cached block's state."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.gdn_cp_deterministic import (
    deterministic_cp_gated_delta_rule,
    deterministic_cp_unavailable_reason,
)

H, HK, D, BLOCK = 24, 8, 128, 1792  # Qwen4Exp GDN at TP2

reason = deterministic_cp_unavailable_reason(D, D)
if reason is not None:
    pytest.skip(f"CP GDN prefill {reason}", allow_module_level=True)

from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import (  # noqa: E402
    l2norm_fwd,
)


def _seq(T, seed, zero_state=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    # Per-head decay from fast to very slow, so the state crosses blocks.
    c = torch.linspace(3, 10, H, device="cuda")
    h0 = (r(H, D, D) * 0.5).bfloat16()
    return dict(
        q=l2norm_fwd(r(T, HK, D).bfloat16()),
        k=l2norm_fwd(r(T, HK, D).bfloat16()),
        v=r(T, H, D).bfloat16(),
        g=F.logsigmoid(r(T, H) * 0.5 + c),
        beta=torch.sigmoid(r(T, H)),
        h0=torch.zeros_like(h0) if zero_state else h0,
    )


def _call(seqs, start=0, stop=None, states=None):
    parts = [{k: v[start:stop] for k, v in s.items() if k != "h0"} for s in seqs]
    x = {k: torch.cat([p[k] for p in parts])[None] for k in parts[0]}
    lens = [p["q"].shape[0] for p in parts]
    cu = torch.tensor([0, *lens], device="cuda").cumsum(0).to(torch.int32)
    h0 = states if states is not None else torch.stack([s["h0"] for s in seqs])
    o, ht = deterministic_cp_gated_delta_rule(
        x["q"], x["k"], x["v"], x["g"], x["beta"], h0, cu, BLOCK
    )
    return o[0], ht, lens


def test_outputs_do_not_depend_on_the_batch():
    a = _seq(6 * BLOCK, 1)
    o_a, h_a, _ = _call([a])
    batches = [
        ([a, _seq(900, 2)], 0),
        ([_seq(3 * BLOCK, 3), a, _seq(200, 4)], 1),
        ([_seq(8 * BLOCK + 300, 5, zero_state=True), a], 1),
    ]
    for seqs, pos in batches:
        o, h, lens = _call(seqs)
        off = sum(lens[:pos])
        assert torch.equal(o[off : off + 6 * BLOCK], o_a)
        assert torch.equal(h[pos], h_a[0])


def test_one_call_equals_block_by_block_and_cached_resume():
    a = _seq(6 * BLOCK - 700, 1)
    o_a, h_a, _ = _call([a])
    # Block by block, the state stored to a bf16 cache in between.
    outs, state = [], a["h0"][None]
    for b in range(6):
        o_b, h_b, _ = _call([a], b * BLOCK, (b + 1) * BLOCK, states=state)
        outs.append(o_b)
        state = h_b.to(torch.bfloat16)
    assert torch.equal(torch.cat(outs), o_a)
    assert torch.equal(state.float(), h_a)
    # Resumed after two cached blocks.
    _, h_2, _ = _call([a], 0, 2 * BLOCK)
    o_r, h_r, _ = _call([a], 2 * BLOCK, None, states=h_2.to(torch.bfloat16))
    assert torch.equal(o_r, o_a[2 * BLOCK :])
    assert torch.equal(h_r, h_a)


def test_close_to_the_fla_kernel():
    a = _seq(4 * BLOCK + 100, 1)
    o, _, _ = _call([a])
    cu = torch.tensor([0, a["q"].shape[0]], device="cuda", dtype=torch.int32)
    ref, _ = chunk_gated_delta_rule(
        a["q"][None],
        a["k"][None],
        a["v"][None],
        a["g"][None],
        a["beta"][None],
        initial_state=a["h0"][None],
        output_final_state=True,
        cu_seqlens=cu,
        state_round_every=BLOCK // 64,
    )
    torch.testing.assert_close(o.float(), ref[0].float(), atol=1e-2, rtol=1e-2)
