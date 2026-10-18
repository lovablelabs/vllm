# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy verification from vocab-parallel logits matches rejection_sample."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.greedy_vocab_parallel import (
    greedy_accept,
    greedy_verify_vocab_parallel,
    shard_block_argmax,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for the greedy verify kernels", allow_module_level=True)

NUM_SPECULATIVE_STEPS = 7
# Two TP shards of a 20000-token vocab padded to 20096: the second shard ends in
# 96 padding columns.
VOCAB_SIZE = 20000
SHARD_WIDTH = 10048
# Logits rows per request: full verification, trimmed by adaptive verification,
# and a bonus-only request.
NUM_LOGITS_PER_REQ = [8, 8, 3, 5, 8, 1, 2, 8]
MAX_NUM_STOP_TOKEN_IDS = 4


def _batch(device: torch.device) -> dict:
    counts = torch.tensor(NUM_LOGITS_PER_REQ, dtype=torch.int32, device=device)
    cu_num_logits = torch.zeros(len(counts) + 1, dtype=torch.int32, device=device)
    cu_num_logits[1:] = torch.cumsum(counts, 0)
    idx_mapping = torch.arange(len(counts), dtype=torch.int32, device=device)
    local_pos = torch.cat(
        [torch.arange(n, dtype=torch.int32, device=device) for n in NUM_LOGITS_PER_REQ]
    )
    return dict(
        cu_num_logits=cu_num_logits,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=torch.repeat_interleave(idx_mapping, counts),
        expanded_local_pos=local_pos,
        pos=100 + local_pos.long(),
    )


def _logits(batch: dict, dtype: torch.dtype) -> torch.Tensor:
    torch.manual_seed(0)
    num_logits = int(batch["cu_num_logits"][-1])
    logits = torch.randn(num_logits, VOCAB_SIZE, device="cuda").to(dtype)
    # Ties across the shard boundary, across the 8192-column blocks of a shard
    # and within a block: the smallest token id wins.
    for row, token_ids in ((0, [9000, 15000]), (1, [100, 9000]), (2, [8191, 8192])):
        logits[row, token_ids] = logits[row].max() + 1
    logits[3, [5, 6]] = logits[3].max() + 1
    return logits


def _min_tokens(batch: dict, logits: torch.Tensor):
    """min_tokens state: request 1 masks its stop tokens on its first three
    rows, request 4 on all of them, and their stop tokens include those rows'
    argmaxes; the other requests have no min_tokens."""
    num_reqs = batch["idx_mapping"].shape[0]
    device = logits.device
    cu = batch["cu_num_logits"].tolist()
    pos = batch["pos"]
    argmax = logits.float().argmax(-1)
    min_lens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    num_stop = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    stop_ids = torch.zeros(
        num_reqs, MAX_NUM_STOP_TOKEN_IDS, dtype=torch.int32, device=device
    )
    min_lens[1] = int(pos[cu[1] + 3]) + 1
    stop_ids[1, :3] = torch.tensor([argmax[cu[1]], argmax[cu[1] + 1], 77])
    num_stop[1] = 3
    min_lens[4] = 10**6
    stop_ids[4, :2] = argmax[cu[4] : cu[4] + 2]
    num_stop[4] = 2
    return min_lens, num_stop, stop_ids


def _masked(batch, logits, min_lens, num_stop, stop_ids) -> torch.Tensor:
    """The logit bias state's min_tokens masking: stop tokens are -inf while
    pos + 1 < min_len."""
    masked = logits.float().clone()
    for row, req in enumerate(batch["expanded_idx_mapping"].tolist()):
        n = int(num_stop[req])
        if n and int(batch["pos"][row]) + 1 < int(min_lens[req]):
            masked[row, stop_ids[req, :n].long()] = float("-inf")
    return masked


def _drafts(batch: dict, masked: torch.Tensor) -> torch.Tensor:
    target = masked.argmax(-1)
    cu = batch["cu_num_logits"].tolist()
    draft_sampled = torch.zeros(masked.shape[0], dtype=torch.int64, device="cuda")
    for req, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        # Accept a request-dependent prefix of drafts, then reject one.
        num_accepted = req % (end - start)
        for i in range(end - start - 1):
            draft = target[start + i]
            draft_sampled[start + i + 1] = (
                draft if i < num_accepted else (draft + 1) % VOCAB_SIZE
            )
    # A placeholder draft (-1) is rejected even where it would match.
    draft_sampled[cu[6] + 1] = -1
    return draft_sampled


def _reference(batch: dict, masked: torch.Tensor, draft_sampled: torch.Tensor):
    num_reqs = batch["idx_mapping"].shape[0]
    return rejection_sample(
        masked.contiguous(),
        None,
        draft_sampled,
        batch["cu_num_logits"],
        batch["pos"],
        batch["idx_mapping"],
        batch["expanded_idx_mapping"],
        batch["expanded_local_pos"],
        torch.zeros(num_reqs, dtype=torch.float32, device="cuda"),
        torch.zeros(num_reqs, dtype=torch.int64, device="cuda"),
        NUM_SPECULATIVE_STEPS,
    )


def _assert_same(actual, expected) -> None:
    sampled, num_sampled = actual
    expected_sampled, expected_num_sampled = expected
    torch.testing.assert_close(num_sampled, expected_num_sampled)
    for req, n in enumerate(expected_num_sampled.tolist()):
        assert sampled[req, :n].tolist() == expected_sampled[req, :n].tolist(), req


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sharded_greedy_verify_matches_rejection_sample(dtype: torch.dtype) -> None:
    batch = _batch(torch.device("cuda"))
    logits = _logits(batch, dtype)
    min_tokens = _min_tokens(batch, logits)
    masked = _masked(batch, logits, *min_tokens)
    draft_sampled = _drafts(batch, masked)
    expected = _reference(batch, masked, draft_sampled)

    # What the all-gather assembles from the two ranks' pairs. The padding
    # columns would win if they were read.
    padding = torch.full(
        (logits.shape[0], 2 * SHARD_WIDTH - VOCAB_SIZE), 1e4, dtype=dtype, device="cuda"
    )
    shards = (
        (logits[:, :SHARD_WIDTH], SHARD_WIDTH, 0),
        (
            torch.cat([logits[:, SHARD_WIDTH:], padding], 1),
            VOCAB_SIZE - SHARD_WIDTH,
            SHARD_WIDTH,
        ),
    )
    pairs = torch.cat(
        [
            shard_block_argmax(
                shard.contiguous(),
                num_valid,
                vocab_start,
                batch["pos"],
                batch["expanded_idx_mapping"],
                *min_tokens,
            )
            for shard, num_valid, vocab_start in shards
        ],
        dim=-1,
    )
    actual = greedy_accept(
        pairs, draft_sampled, batch["cu_num_logits"], NUM_SPECULATIVE_STEPS
    )
    _assert_same(actual, expected)


def test_unsharded_greedy_verify_matches_rejection_sample() -> None:
    batch = _batch(torch.device("cuda"))
    logits = _logits(batch, torch.bfloat16)
    min_tokens = _min_tokens(batch, logits)
    masked = _masked(batch, logits, *min_tokens)
    draft_sampled = _drafts(batch, masked)
    expected = _reference(batch, masked, draft_sampled)
    actual = greedy_verify_vocab_parallel(
        logits,
        VOCAB_SIZE,
        0,
        1,
        draft_sampled,
        batch["cu_num_logits"],
        batch["pos"],
        batch["expanded_idx_mapping"],
        *min_tokens,
        NUM_SPECULATIVE_STEPS,
    )
    _assert_same(actual, expected)


def test_nan_logit_never_wins() -> None:
    device = torch.device("cuda")
    logits = torch.randn(2, VOCAB_SIZE, device=device)
    best = logits.argmax(-1)
    logits[0, (best[0] + 1) % VOCAB_SIZE] = float("nan")
    logits[1, best[1]] = float("nan")
    expected = logits.nan_to_num(nan=float("-inf")).argmax(-1)
    zeros = torch.zeros(2, dtype=torch.int32, device=device)
    sampled, num_sampled = greedy_verify_vocab_parallel(
        logits,
        VOCAB_SIZE,
        0,
        1,
        torch.zeros(2, dtype=torch.int64, device=device),
        torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        torch.zeros(2, dtype=torch.int64, device=device),
        torch.arange(2, dtype=torch.int32, device=device),
        zeros,
        zeros,
        torch.zeros(2, MAX_NUM_STOP_TOKEN_IDS, dtype=torch.int32, device=device),
        NUM_SPECULATIVE_STEPS,
    )
    assert num_sampled.tolist() == [1, 1]
    assert sampled[:, 0].tolist() == expected.tolist()
