# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic serving: QSA decode rows give the same bits whichever program
layout and warp count the batch size picks (fixed fold tree of per-leaf
partials), including partial selections and rows without a request."""

import pytest
import torch

import vllm.models.qwen4_exp.nvidia.ops.qsa_tree as qsa_tree
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("QSA kernels need CUDA", allow_module_level=True)

QUERY_HEADS, HEAD_DIM, PAGE, SEQ = 12, 256, 1792, 16384


def _inputs(rows: int, topk: int):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    TOPK = topk
    requests = max(1, rows // 4)
    pages = SEQ // PAGE + 1
    shape = (requests * pages, PAGE, 1, HEAD_DIM)
    k = (torch.randn(shape, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(shape, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    table = torch.arange(requests * pages, device="cuda", dtype=torch.int32)
    table = table.view(requests, pages)
    token_to_req = (torch.arange(rows, device="cuda") // 4).to(torch.int32)
    token_to_req[rows // 2] = -1  # a row without a request
    indices = torch.empty(rows, TOPK + 1, device="cuda", dtype=torch.int32)
    for row in range(rows):
        perm = torch.randperm(SEQ, device="cuda", generator=gen)[:TOPK]
        indices[row, :TOPK] = perm.sort().values.to(torch.int32)
        count = (TOPK, TOPK - 37, 2049, 1500, 129, 1, 0)[row % 7]  # partial leaves too
        indices[row, count:TOPK] = -1
        indices[row, TOPK] = count
    q = torch.randn(rows, QUERY_HEADS, HEAD_DIM, device="cuda", generator=gen)
    gate = torch.randn(rows, QUERY_HEADS, HEAD_DIM, device="cuda", generator=gen)
    return q.bfloat16(), k, v, indices, table, token_to_req, gate.bfloat16()


def _attend(x, monkeypatch, deterministic: bool, layout=None):
    q, k, v, indices, table, token_to_req, gate = x
    monkeypatch.setattr(qsa_ops, "deterministic_serving", lambda: deterministic)
    if layout is not None:
        monkeypatch.setattr(qsa_tree, "qsa_tree_layout", lambda *args: layout)
    try:
        return qsa_ops.qsa_sparse_paged_attention(
            q, k, v, indices, table, token_to_req, False,
            k_scale=1.0, v_scale=1.0, output_gate=gate,
        )  # fmt: skip
    finally:
        monkeypatch.undo()


# 2051: Qwen4Exp's 2048-token budget plus the compressed-block expansion's three.
@pytest.mark.parametrize("topk", [2048, 2051])
@pytest.mark.parametrize("rows", [4, 32, 96])
def test_deterministic_qsa_tree_bits_do_not_depend_on_layout(rows, topk, monkeypatch):
    x = _inputs(rows, topk)
    ref = _attend(x, monkeypatch, True, layout=(16, 4))
    assert torch.isfinite(ref).all()
    for layout in ((16, 2), (16, 8), (4, 1), (4, 2), (4, 4), (4, 8)):
        assert torch.equal(_attend(x, monkeypatch, True, layout=layout), ref), layout
    # The served layout for this row count, and the rows' bits at other row
    # counts (a prefix of the batch).
    assert torch.equal(_attend(x, monkeypatch, True), ref)
    head = tuple(t[:4] if t.dim() and t.shape[0] == rows else t for t in x)
    assert torch.equal(_attend(head, monkeypatch, True), ref[:4])
    # Same result as the default kernels up to rounding.
    default = _attend(x, monkeypatch, False)
    torch.testing.assert_close(ref.float(), default.float(), atol=2e-2, rtol=2e-2)
