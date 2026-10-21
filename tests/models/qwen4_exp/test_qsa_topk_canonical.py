# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA block selection must be reproducible: the exact top-k with ties at the
k-th score resolved to the lowest columns, ascending, on every call."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
    _TOPK_WORKSPACE_BYTES,
    _canonicalize_expand,
    _canonicalize_topk,
    _topk,
    expand_qsa_block_indices,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="requires a CUDA device"
)
DEVICE = "cuda"
COMPRESS_RATIO = 4
BLOCK_TOPK = 512


def _ordered_key(logits):
    """The ops' score order as int32: float order with -0.0 below +0.0."""
    bits = logits.view(torch.int32)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


def _reference(logits, visible, k):
    cols = torch.arange(logits.shape[1], device=logits.device)
    hidden = cols[None, :] >= visible[:, None]
    keys = _ordered_key(logits).masked_fill(hidden, torch.iinfo(torch.int32).min)
    # Stable sort on the key keeps tied columns in ascending order.
    top = torch.sort(keys, dim=1, descending=True, stable=True).indices[:, :k]
    top = top.to(torch.int32).masked_fill(hidden.gather(1, top), 2**30)
    top = torch.sort(top, dim=1).values
    return top.masked_fill(top == 2**30, -1)


def _logits(rows, width, ties, seed):
    torch.manual_seed(seed)
    logits = torch.randn(rows, width, device=DEVICE)
    if ties:
        logits = (logits * 2).round() / 2  # many exact ties at the k-th score
    visible = torch.randint(300, width + 1, (rows,), device=DEVICE, dtype=torch.int32)
    visible[0] = 100  # fewer visible blocks than k
    return logits, visible


# 48 rows take the cooperative op, 96 and 512 the persistent one.
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("rows", [48, 96, 512])
def test_topk_matches_lowest_column_reference(rows, ties):
    width = 4096
    logits, visible = _logits(rows, width, ties, seed=0)
    workspace = torch.empty(_TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device=DEVICE)
    outs = []
    for _ in range(3):
        out = torch.empty(rows, BLOCK_TOPK, device=DEVICE, dtype=torch.int32)
        _topk(
            logits,
            visible,
            BLOCK_TOPK * COMPRESS_RATIO,
            COMPRESS_RATIO,
            out,
            workspace,
        )
        outs.append(out)
    assert all(torch.equal(outs[0], out) for out in outs[1:])
    assert torch.equal(outs[0], _reference(logits, visible, BLOCK_TOPK))


def _raw_topk(logits, visible, rescan):
    """The radix-select result and its tie flags; with rescan, every row is
    flagged so the canonicalization re-derives it from the logits."""
    rows, width = logits.shape
    workspace = torch.empty(_TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device=DEVICE)
    raw = torch.empty(rows, BLOCK_TOPK, device=DEVICE, dtype=torch.int32)
    flags = torch.empty(rows, device=DEVICE, dtype=torch.int32)
    torch.ops._C.persistent_topk(
        logits, visible, raw, workspace, BLOCK_TOPK, width, flags
    )
    if rescan:
        flags.fill_(1)
    return raw, flags


@pytest.mark.parametrize("rescan", [False, True])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("rows", [48, 512])
def test_canonicalize_scan_shape_keeps_the_output(rows, ties, rescan):
    width = 12288
    logits, visible = _logits(rows, width, ties, seed=1)
    raw, flags = _raw_topk(logits, visible, rescan)
    outs = []
    for config in (None, (512, 1, 1), (2048, 4, 1), (2048, 8, 4), (8192, 8, 2)):
        out = raw.clone()
        _canonicalize_topk(logits, visible, BLOCK_TOPK, out, flags, config=config)
        outs.append(out)
    assert all(torch.equal(outs[0], out) for out in outs[1:])
    assert torch.equal(outs[0], _reference(logits, visible, BLOCK_TOPK))


@pytest.mark.parametrize("rescan", [False, True])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("rows", [16, 96, 256])
def test_canonicalize_expand_matches_canonicalize_then_expand(rows, ties, rescan):
    """The decode path's fused kernel writes the same packed token indices
    (expanded blocks, causal tail, trailing count) as the two kernels."""
    width = 8192
    logits, _ = _logits(rows, width, ties, seed=2)
    positions = torch.randint(0, 4 * width, (rows,), device=DEVICE)
    positions[0] = 37  # fewer visible blocks than k, open tail group
    positions[1] = 4 * BLOCK_TOPK + 3  # exactly k complete blocks, full tail
    visible = ((positions + 1) // COMPRESS_RATIO).clamp(max=width).to(torch.int32)
    raw, flags = _raw_topk(logits, visible, rescan)
    packed_width = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO
    expected = torch.full((rows, packed_width), -7, device=DEVICE, dtype=torch.int32)
    blocks = raw.clone()
    _canonicalize_topk(logits, visible, BLOCK_TOPK, blocks, flags)
    expand_qsa_block_indices(
        blocks,
        positions,
        visible,
        COMPRESS_RATIO,
        BLOCK_TOPK * COMPRESS_RATIO,
        expected,
    )
    actual = torch.full_like(expected, -7)
    _canonicalize_expand(
        logits, visible, raw.clone(), flags, positions, COMPRESS_RATIO, actual
    )
    assert torch.equal(actual, expected)
