# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy verification from vocab-parallel logits.

A greedy request accepts a draft token iff it equals the target argmax, and its
bonus or corrected token is the target argmax. When every request of a
speculative batch is greedy and needs no logits processing but min_tokens (whose
stop-token masking the shard argmax applies), each TP rank finds the (max,
first argmax) of every 8192-column block of its vocab shard, the ranks
exchange those pairs, and one kernel walks the drafts. That skips the full-vocab
logits all-gather, its layout copy and the generic sampler's float32 copy and
per-block statistics.

For finite logits the result is rejection_sample's at temperature 0: the token
with the largest logit, the smallest such id on ties (blocks are scanned in
vocab order, ranks in shard order). A NaN logit never wins; rows holding one
may differ from the generic sampler, which drops the whole block around it.
"""

import torch

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.triton_utils import tl, triton

# Vocab columns per program of the shard argmax.
_BLOCK = 8192


@triton.jit
def _shard_block_argmax_kernel(
    # [num_logits, 2 * num_blocks] float32: per block (max, token id bit-cast)
    out_ptr,
    out_stride,
    # [num_logits, shard_width]
    logits_ptr,
    logits_stride,
    num_valid,
    vocab_start,
    # [num_logits]
    pos_ptr,
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    min_lens_ptr,
    num_stop_token_ids_ptr,
    # [max_num_reqs, max_num_stop_token_ids]
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    cols = block_idx * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        logits_ptr + row * logits_stride + cols,
        mask=cols < num_valid,
        other=float("-inf"),
    ).to(tl.float32)
    # min_tokens masks the request's stop tokens while pos + 1 < min_len, as
    # the logit bias state does.
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    pos = tl.load(pos_ptr + row)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    num_stop = tl.load(num_stop_token_ids_ptr + req_state_idx)
    num_stop = tl.where(pos + 1 < min_len, num_stop, 0)
    for k in range(num_stop):
        stop_col = (
            tl.load(stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + k)
            - vocab_start
        )
        values = tl.where(cols == stop_col, float("-inf"), values)
    values = tl.where(values != values, float("-inf"), values)
    value, idx = tl.max(values, axis=0, return_indices=True)
    token_id = (block_idx * BLOCK + idx + vocab_start).to(tl.int32)
    tl.store(out_ptr + row * out_stride + 2 * block_idx, value)
    tl.store(
        out_ptr + row * out_stride + 2 * block_idx + 1,
        token_id.to(tl.float32, bitcast=True),
    )


@triton.jit
def _greedy_accept_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_logits, 2 * NUM_BLOCKS]: blocks in vocab order
    pairs_ptr,
    pairs_stride,
    # [num_logits]
    draft_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    NUM_BLOCKS: tl.constexpr,
    PADDED_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    blocks = tl.arange(0, PADDED_NUM_BLOCKS)
    blocks_mask = blocks < NUM_BLOCKS

    accepted_length = tl.zeros((), tl.int32)
    verifying = True
    for i in range(end_idx - start_idx):
        if verifying:
            logit_idx = start_idx + i
            values = tl.load(
                pairs_ptr + logit_idx * pairs_stride + 2 * blocks,
                mask=blocks_mask,
                other=float("-inf"),
            )
            # The first block holding the maximum: the smallest token id.
            best_block = tl.argmax(values, axis=0)
            target = (
                tl.load(pairs_ptr + logit_idx * pairs_stride + 2 * best_block + 1)
                .to(tl.int32, bitcast=True)
                .to(tl.int64)
            )
            # The bonus row has no draft; a -1 placeholder draft is rejected.
            has_draft = logit_idx < end_idx - 1
            draft = tl.load(
                draft_sampled_ptr + logit_idx + 1, mask=has_draft, other=-1
            ).to(tl.int64)
            accepted = has_draft & (draft >= 0) & (draft == target)
            tl.store(
                sampled_ptr + req_idx * sampled_stride + i,
                tl.where(accepted, draft, target),
            )
            accepted_length += accepted.to(tl.int32)
            verifying = accepted
    tl.store(num_sampled_ptr + req_idx, accepted_length + 1)


def shard_block_argmax(
    local_logits: torch.Tensor,
    num_valid: int,
    vocab_start: int,
    pos: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Per 8192-column block of this rank's ``[num_logits, shard]`` logits, the
    (max, first argmax) over its first ``num_valid`` columns, which hold token
    ids ``vocab_start ..``, with the min_tokens stop-token masking applied:
    ``[num_logits, 2 * num_blocks]`` float32 with the token id bit-cast from
    int32."""
    assert local_logits.ndim == 2 and local_logits.stride(-1) == 1
    num_logits = local_logits.shape[0]
    num_blocks = triton.cdiv(local_logits.shape[1], _BLOCK)
    pairs = torch.empty(
        num_logits, 2 * num_blocks, dtype=torch.float32, device=local_logits.device
    )
    if num_logits > 0:
        _shard_block_argmax_kernel[(num_logits, num_blocks)](
            pairs,
            pairs.stride(0),
            local_logits,
            local_logits.stride(0),
            num_valid,
            vocab_start,
            pos,
            expanded_idx_mapping,
            min_lens,
            num_stop_token_ids,
            stop_token_ids,
            stop_token_ids.stride(0),
            BLOCK=_BLOCK,
            num_warps=8,
        )
    return pairs


def greedy_accept(
    pairs: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Walk the drafts against the target argmax of every logits row, given the
    block (max, token id) pairs of the whole vocab in order. Returns
    ``(sampled [num_reqs, K + 1] int64, num_sampled [num_reqs] int32)`` as
    rejection_sample does."""
    num_reqs = cu_num_logits.shape[0] - 1
    num_blocks = pairs.shape[1] // 2
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    _greedy_accept_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        pairs,
        pairs.stride(0),
        draft_sampled,
        cu_num_logits,
        NUM_BLOCKS=num_blocks,
        PADDED_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        num_warps=1,
    )
    return sampled, num_sampled


def greedy_verify_vocab_parallel(
    local_logits: torch.Tensor,
    num_valid: int,
    vocab_start: int,
    tp_size: int,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """rejection_sample's ``(sampled, num_sampled)`` for an all-greedy batch,
    from this rank's vocab shard of the target logits."""
    pairs = shard_block_argmax(
        local_logits,
        num_valid,
        vocab_start,
        pos,
        expanded_idx_mapping,
        min_lens,
        num_stop_token_ids,
        stop_token_ids,
    )
    if tp_size > 1:
        # Every rank has the same number of blocks, so the gathered pairs keep
        # the vocab order.
        pairs = tensor_model_parallel_all_gather(pairs, dim=-1)
    return greedy_accept(pairs, draft_sampled, cu_num_logits, num_speculative_steps)
