# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling under VLLM_DETERMINISTIC_SERVING: a step carries at most
VLLM_DETERMINISTIC_MAX_PREFILLS prefill chunks (with any number of decodes),
prefill chunks end at KV block boundaries, and the prefix cache keeps
prompt-token blocks only."""

import random

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
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
@pytest.mark.parametrize("seed", range(4))
def test_prefill_chunks_are_capped_and_block_aligned(
    monkeypatch, max_prefills, async_scheduling, seed
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
                # Ends at or before the next block boundary.
                assert n <= BLOCK_SIZE - start % BLOCK_SIZE
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
