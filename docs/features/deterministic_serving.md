# Deterministic Serving

With `VLLM_DETERMINISTIC_SERVING=1`, a request's greedy output does not depend on what else the
server is doing: how many requests run concurrently, what is in the prefix cache, or which boot of
the server answers. Sending the same request again returns the same tokens.

[Batch invariance](batch_invariance.md) (`VLLM_BATCH_INVARIANT=1`) gives the same guarantee for
any model, but replaces kernels wholesale and costs ~2.5x decode throughput. Deterministic serving
keeps the fast kernels and fixes only what changes numerics in a serving loop.

## Usage

```bash
VLLM_DETERMINISTIC_SERVING=1 vllm serve <model> ...
```

On GDN models (Qwen3-Next, Qwen4Exp), pick a KV block size that is a multiple of 64, for example
`--block-size 1792` instead of 1776: prefill chunks can then span whole blocks (see below), which
cuts the time to first token of a cold prompt by about 30% compared with one block per chunk.

On Qwen4Exp with MTP, `VLLM_QWEN4_EXP_MTP_FP8_HEAD=1` also computes the drafter's logits from an
FP8 copy of the lm_head shard (one more byte per weight per GPU): the draft step's largest GEMM
takes 52 instead of 92 us on B200, and a row keeps its bits at any batch size. It changes the
draft tokens, not the target's verification.

## What it changes

Scheduling:

- Prefill chunks start and end at KV block boundaries and are never cut short inside a block by
  the token budget, so a block's KV and recurrent state depend only on the tokens before it,
  whether a cold prefill or a prefix-cache hit produces them.
- A chunk covers as many whole blocks as the token budget allows on GDN models when the block
  size is a multiple of the FLA prefill kernel's 64-token chunk, the GDN state cache is bf16 (or
  fp32) and any other recurrent layer is a short convolution: the kernel rounds its carried state
  to bf16 at the inner block boundaries, exactly as storing it to the cache between one-block
  chunks would. Otherwise each chunk is one block.
- Only blocks made of prompt tokens are cached. Blocks holding output tokens were computed by
  decode kernels; a later turn that re-sends them prefills them as a cold server would.
- `VLLM_DETERMINISTIC_MAX_PREFILLS` caps the prefill chunks per step (default 0: no cap). Decode
  and prefill rows still share steps.

Kernels:

- cuBLASLt without split-K: each output row is the same at any row count.
- The batch-invariant `mean`, `softmax` and `log_softmax` kernels for ops that run eagerly.
- GDN prefill, on SM100 with whole-block chunks: FlashInfer's chunk-parallel kernels with their
  segments set to the KV block and the sequential fixup between segments replaced by one that
  rounds each block-end state to bf16, as the state cache does; a sequence's outputs then depend
  only on its own tokens and the block boundaries. Elsewhere the FLA Triton kernel (per
  sequence), autotuned at fixed warp counts: the warp count changes the bits of four of its
  kernels, and autotuning picks by timing (often between near-equal timings), so a server could
  otherwise answer with other bits than the last one. Tile sizes and pipeline stages stay
  autotuned; at the fixed counts they leave the bits unchanged.
- Qwen4Exp: QSA sparse attention on fixed split configs, with the decode and prefill rows of a
  mixed step launched separately; decode rows give each 128-column leaf of the selection its own
  softmax partial and fold the partials in one fixed tree, so the program count follows the batch
  size without changing the bits (the softmax row sums go through the tensor cores, so the warp
  count does not change them either); instead of the per-row-count skinny GEMM plans, a TMA GEMM with
  cuBLASLt's bits at a few rows; the HC down projection on the K-sliced GEMM at every row count,
  in one clustered launch at a few rows; the MTP drafter's two pre-norms in one kernel each (the
  NVIDIA model runs eagerly, where the native norm is eight to ten kernels).

## Measured

Qwen4Exp NVFP4 with MTP (3 speculative tokens), TP2 x 2 replicas on B200, block size 1792, 80
agent prompts of 27-95k tokens, greedy:

- identical outputs at c1, c2, c16 and c64, cold vs cached, across boots that each autotune the
  Triton kernels and the MoE on their own or load a saved autotune cache, and for turn 2 of a
  conversation on a warm vs a fresh server;
- vs the default kernels in the same session, both with `VLLM_QWEN4_EXP_MTP_FP8_HEAD=1`: output
  throughput -1.5% at c2 (cached) and +3% cold at c2 (time to first token -9%, p99 1.5 s vs 5.0 s),
  -2 to -4% at c16 and c64 against the mean of two default runs (which differ by 5-9% from each
  other; the deterministic runs agree within 0.1%); single-request TPOT +2% (1.87 vs 1.83 ms). The
  remaining decode cost is the GEMMs and the HC down projection at one or two requests per
  replica, where the default kernels' per-row-count plans are 1-2 us faster per call, and the
  QSA fold tree's 0.4-0.8% at 12-32 requests per replica.

## Limits

- Validated at TP2. A two-rank all-reduce is order-free; TP >= 4 also needs a one-stage
  all-reduce with a fixed reduction order (`VLLM_CUSTOM_ALLREDUCE_ALGO=1stage`), not tested here.
- Validated on B200. The GDN prefill warp counts are fixed on every GPU, but that the tile and
  stage choices left to autotuning keep the bits was checked on B200 only;
  `tests/kernels/mamba/test_gdn_autotune_bits.py` checks it on the GPU it runs on. The
  chunk-parallel GDN prefill uses private FlashInfer 0.7 functions and falls back to the FLA kernel
  when they are missing (`tests/kernels/mamba/test_gdn_cp_deterministic.py`). The FlashInfer
  TRTLLM FP4 MoE is left to its autotuner: every tactic gives a token the same bits at any batch
  size at Qwen4Exp's MoE shape on B200 (`test_trtllm_fp4_moe_tactics_keep_token_bits`).
- Prefix-cache hits must land on whole KV blocks (hash block size equal to the block size).
- A request preempted by recompute re-prefills its output tokens and may diverge from then on.
- Logprobs of a scored prompt (prefill) and of the same tokens generated (decode) still come
  from different kernels and are not bit-equal.
