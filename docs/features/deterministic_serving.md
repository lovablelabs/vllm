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
roughly halves the cold-prefill cost per token compared with one block per chunk.

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
- GDN prefill on the FLA Triton kernel (per sequence) instead of FlashInfer's chunk-parallel one.
- Qwen4Exp: QSA sparse attention on fixed split configs, with the decode and prefill rows of a
  mixed step launched separately; instead of the per-row-count skinny GEMM plans, a TMA GEMM with
  cuBLASLt's bits at a few rows; the HC down projection on the K-sliced GEMM at every row count,
  in one clustered launch at a few rows.

## Measured

Qwen4Exp NVFP4 with MTP (3 speculative tokens), TP2 x 2 replicas on B200, 80 agent prompts of
27-95k tokens, greedy:

- identical outputs at c1, c2, c16, c32, c64 and c128, cold vs cached, two boots, and turn 2 of a
  conversation on a warm vs a fresh server;
- throughput vs the default kernels: about -5% at c16 and -6 to -8% at c64-c128; single-request
  TPOT +15% (the skinny GEMM plans are 2-3 us faster per GEMM at one or two requests); cold
  prefill TTFT about +50%.

## Limits

- Validated at TP2. A two-rank all-reduce is order-free; TP >= 4 also needs a one-stage
  all-reduce with a fixed reduction order (`VLLM_CUSTOM_ALLREDUCE_ALGO=1stage`), not tested here.
- Prefix-cache hits must land on whole KV blocks (hash block size equal to the block size).
- A request preempted by recompute re-prefills its output tokens and may diverge from then on.
- Logprobs of a scored prompt (prefill) and of the same tokens generated (decode) still come
  from different kernels and are not bit-equal.
