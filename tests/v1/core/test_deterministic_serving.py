# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling under VLLM_DETERMINISTIC_SERVING: a step carries at most
VLLM_DETERMINISTIC_MAX_PREFILLS prefill chunks (with any number of decodes),
prefill chunks start and end at KV block boundaries (one block each, or whole
blocks up to the budget on GDN models), and the prefix cache keeps prompt-token
blocks only."""

import random

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.core.sched.scheduler import deterministic_multiblock_prefill
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_requests, create_scheduler

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def deterministic_serving(monkeypatch):
    monkeypatch.setenv("VLLM_DETERMINISTIC_SERVING", "1")


def _output(scheduler_output, sampled: dict[str, bool]) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[7] if sampled[req_id] else [] for req_id in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize("max_prefills", [0, 1, 2])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("multiblock", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_prefill_chunks_are_capped_and_block_aligned(
    monkeypatch, max_prefills, async_scheduling, multiblock, seed
):
    monkeypatch.setenv("VLLM_DETERMINISTIC_MAX_PREFILLS", str(max_prefills))
    random.seed(seed)
    scheduler = create_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=256,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu" if async_scheduling else None,
        enable_prefix_caching=True,
        async_scheduling=async_scheduling,
        block_size=BLOCK_SIZE,
    )
    # The attention-only test model gets one block per chunk; force the GDN-model
    # behaviour (whole blocks up to the budget) to check it too.
    scheduler.deterministic_multiblock_prefill = multiblock
    pending = [
        create_requests(
            1,
            num_tokens=random.choice([5, 40, 300, 700]),
            max_tokens=random.randint(3, 30),
            req_ids=[f"r{i}"],
        )[0]
        for i in range(12)
    ]
    in_flight = []
    max_decode_batch = 0
    mixed_steps = 0
    max_chunk = 0
    for _ in range(4000):
        if pending and random.random() < 0.3:
            scheduler.add_request(pending.pop(0))
        output = scheduler.schedule()
        new = {r.req_id for r in output.scheduled_new_reqs}
        prefill_rows = []
        for req_id, n in output.num_scheduled_tokens.items():
            request = scheduler.requests[req_id]
            start = request.num_computed_tokens - n
            if req_id in new or start < request.num_prompt_tokens:
                prefill_rows.append(req_id)
                max_chunk = max(max_chunk, n)
                end = start + n
                # Starts at a block boundary and ends at one or at the prompt end.
                assert start % BLOCK_SIZE == 0
                assert end % BLOCK_SIZE == 0 or end == request.num_prompt_tokens
                if not multiblock:
                    assert n <= BLOCK_SIZE
        assert not max_prefills or len(prefill_rows) <= max_prefills
        decode_rows = len(output.num_scheduled_tokens) - len(prefill_rows)
        max_decode_batch = max(max_decode_batch, decode_rows)
        mixed_steps += bool(prefill_rows and decode_rows)
        # A step samples a token for rows that reach the end of their prompt.
        sampled = {
            req_id: scheduler.requests[req_id].num_computed_tokens
            >= scheduler.requests[req_id].num_tokens
            for req_id in output.num_scheduled_tokens
        }
        in_flight.append((output, sampled))
        depth = 2 if async_scheduling else 1
        if len(in_flight) == depth or not output.num_scheduled_tokens:
            done, done_sampled = in_flight.pop(0)
            scheduler.update_from_output(done, _output(done, done_sampled))
        if not pending and not in_flight and not scheduler.has_unfinished_requests():
            break
    assert not pending and not scheduler.has_unfinished_requests()
    assert max_decode_batch > 1
    assert mixed_steps > 0
    assert (max_chunk > BLOCK_SIZE) == multiblock


def _group(spec):
    return KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=spec)


def _gdn(block_size, ssm_dtype=torch.bfloat16, mamba_type=None):
    return MambaSpec(
        block_size=block_size,
        shapes=((4, 8), (2, 8, 8)),
        dtypes=(torch.bfloat16, ssm_dtype),
        mamba_type=mamba_type or MambaAttentionBackendEnum.GDN_ATTN,
    )


@pytest.mark.parametrize(
    "groups, block_size, expected",
    [
        # Attention only: one block per chunk.
        (
            [
                FullAttentionSpec(
                    block_size=1792, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
                )
            ],
            1792,
            False,
        ),
        # GDN with a 64-aligned block and a bf16 or fp32 state: whole blocks.
        ([_gdn(1792)], 1792, True),
        ([_gdn(1792, torch.float32)], 1792, True),
        # Block size not a multiple of the FLA chunk, fp16 state, other mamba.
        ([_gdn(1776)], 1776, False),
        ([_gdn(1792, torch.float16)], 1792, False),
        ([_gdn(1792, mamba_type=MambaAttentionBackendEnum.MAMBA2)], 1792, False),
        # Short convolutions next to GDN layers keep their inputs as they are;
        # on their own there is no GDN layer to batch blocks for.
        (
            [_gdn(1792), _gdn(1792, mamba_type=MambaAttentionBackendEnum.SHORT_CONV)],
            1792,
            True,
        ),
        ([_gdn(1792, mamba_type=MambaAttentionBackendEnum.SHORT_CONV)], 1792, False),
    ],
)
def test_multiblock_prefill_needs_gdn_and_aligned_blocks(groups, block_size, expected):
    config = KVCacheConfig(
        num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[_group(g) for g in groups]
    )
    assert deterministic_multiblock_prefill(config, block_size) == expected


@pytest.mark.parametrize("deterministic", [True, False])
def test_prefix_cache_keeps_prompt_blocks_only(monkeypatch, deterministic):
    if not deterministic:
        monkeypatch.setenv("VLLM_DETERMINISTIC_SERVING", "0")
    scheduler = create_scheduler(
        enable_prefix_caching=True, block_size=BLOCK_SIZE, max_num_batched_tokens=256
    )
    (first,) = create_requests(1, num_tokens=40, max_tokens=30, ignore_eos=True)
    scheduler.add_request(first)
    while scheduler.has_unfinished_requests():
        output = scheduler.schedule()
        sampled = {
            req_id: scheduler.requests[req_id].num_computed_tokens
            >= scheduler.requests[req_id].num_tokens
            for req_id in output.num_scheduled_tokens
        }
        scheduler.update_from_output(output, _output(output, sampled))
    # Turn two re-sends the first request's prompt and output tokens.
    params = SamplingParams(max_tokens=4)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    second = Request(
        request_id="turn2",
        prompt_token_ids=list(first.prompt_token_ids) + [7] * 30 + [3] * 5,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    scheduler.add_request(second)
    output = scheduler.schedule()
    num_hit = second.num_computed_tokens - output.num_scheduled_tokens["turn2"]
    if deterministic:
        # Only the two full prompt blocks hit; the blocks holding the first
        # request's output tokens are recomputed, one block per step.
        assert num_hit == 2 * BLOCK_SIZE
        assert output.num_scheduled_tokens["turn2"] == BLOCK_SIZE
    else:
        assert num_hit == 4 * BLOCK_SIZE
