# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic serving: the QSA split kernel (prefill rows) gives the same
bits at every warp count."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("QSA kernels need CUDA", allow_module_level=True)

QUERY_HEADS, HEAD_DIM, TOPK, PAGE, SEQ = 12, 256, 2048, 1792, 16384


def _inputs(rows: int):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    requests = max(1, rows // 4)
    pages = SEQ // PAGE + 1
    shape = (requests * pages, PAGE, 1, HEAD_DIM)
    k = (torch.randn(shape, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(shape, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
    table = torch.arange(requests * pages, device="cuda", dtype=torch.int32)
    table = table.view(requests, pages)
    token_to_req = (torch.arange(rows, device="cuda") // 4).to(torch.int32)
    indices = torch.empty(rows, TOPK + 1, device="cuda", dtype=torch.int32)
    for row in range(rows):
        perm = torch.randperm(SEQ, device="cuda", generator=gen)[:TOPK]
        indices[row, :TOPK] = perm.sort().values.to(torch.int32)
    count = TOPK - 37 * (rows % 3)  # a partly filled last tile
    indices[:, count:TOPK] = -1
    indices[:, TOPK] = count
    q = torch.randn(rows, QUERY_HEADS, HEAD_DIM, device="cuda", generator=gen)
    gate = torch.randn(rows, QUERY_HEADS, HEAD_DIM, device="cuda", generator=gen)
    return q.bfloat16(), k, v, indices, table, token_to_req, gate.bfloat16()


def _attend(x, monkeypatch, deterministic: bool, warps: int | None = None):
    q, k, v, indices, table, token_to_req, gate = x
    monkeypatch.setattr(qsa_ops, "deterministic_serving", lambda: deterministic)
    select = qsa_ops._select_config
    if warps is not None:

        def forced(*args, **kwargs):
            block_n, _, tiles, splits = select(*args, **kwargs)
            return block_n, warps, tiles, splits

        monkeypatch.setattr(qsa_ops, "_select_config", forced)
    try:
        return qsa_ops.qsa_sparse_paged_attention(
            q, k, v, indices, table, token_to_req, True,
            k_scale=1.0, v_scale=1.0, output_gate=gate,
        )  # fmt: skip
    finally:
        monkeypatch.setattr(qsa_ops, "_select_config", select)


@pytest.mark.parametrize("rows", [4, 32])
def test_deterministic_qsa_bits_do_not_depend_on_warps(rows, monkeypatch):
    x = _inputs(rows)
    ref = _attend(x, monkeypatch, True, warps=1)
    for warps in (2, 4, 8):
        assert torch.equal(_attend(x, monkeypatch, True, warps=warps), ref), warps
    # Same result as the default kernels up to rounding.
    default = _attend(x, monkeypatch, False)
    torch.testing.assert_close(ref.float(), default.float(), atol=2e-2, rtol=2e-2)
