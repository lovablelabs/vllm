# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One FLA GDN prefill call over several KV blocks with ``state_round_every``
equals one call per block with the state stored to a bf16 cache in between,
bit for bit (deterministic serving's multi-block prefill chunks)."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FLA GDN kernels need CUDA", allow_module_level=True)

from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops.utils import (  # noqa: E402
    FLA_CHUNK_SIZE,
)

BLOCK = 1792
H, HK, DK, DV = 16, 8, 128, 128  # value heads, key heads (GQA), head dims


def _inputs(T):
    q = F.normalize(torch.randn(1, T, HK, DK, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, T, HK, DK, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(1, T, H, DV, device="cuda").bfloat16()
    g = (-torch.rand(1, T, H, device="cuda") * 0.1).float()
    beta = torch.rand(1, T, H, device="cuda").bfloat16()
    return q, k, v, g, beta


def _call(seqs, states, round_every):
    cat = [torch.cat([s[i] for s in seqs], dim=1) for i in range(5)]
    lens = torch.tensor([0] + [s[0].shape[1] for s in seqs]).cumsum(0)
    o, ht = chunk_gated_delta_rule(
        *cat,
        initial_state=torch.stack(states).float(),
        output_final_state=True,
        cu_seqlens=lens.to(device="cuda", dtype=torch.int32),
        state_round_every=round_every,
    )
    return o, ht


@pytest.mark.parametrize(
    "lengths", [(2 * BLOCK,), (4 * BLOCK + 300,), (1000, 3 * BLOCK), (300, 6 * BLOCK)]
)
def test_multi_block_call_matches_per_block_calls(lengths):
    torch.manual_seed(0)
    seqs = [_inputs(T) for T in lengths]
    states = [(torch.randn(H, DV, DK, device="cuda") * 0.1).bfloat16() for _ in seqs]
    o, ht = _call(seqs, states, BLOCK // FLA_CHUNK_SIZE)

    ref_o, ref_h = [], []
    for seq, state, T in zip(seqs, states, lengths):
        outs = []
        for start in range(0, T, BLOCK):
            block = [t[:, start : start + BLOCK] for t in seq]
            o_i, h_i = _call([block], [state], 0)
            outs.append(o_i)
            state = h_i[0].to(torch.bfloat16)  # the bf16 state cache
        ref_o.append(torch.cat(outs, dim=1))
        ref_h.append(state)
    assert torch.equal(o, torch.cat(ref_o, dim=1))
    for i, state in enumerate(ref_h):
        assert torch.equal(ht[i].to(torch.bfloat16), state)
