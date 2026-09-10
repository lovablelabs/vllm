# Historical QSA query-packing backport

This patch targets the scorer at commit
`80389cfedd5040e382d64a64b1782f66de1a38bf`, immediately before
[upstream #54513](https://github.com/vllm-project/vllm/pull/54513).
That September 2, 2026 change already introduced a tiled prefill scorer on
current main. **These measurements compare against the historical scorer,
not current main. No novelty or current-main performance claim is made.**
The base's `vllm/models/qwen4_exp/nvidia/ops/qsa.py` has the same executable AST
as the measured original after removing its module docstring.

[results.json](results.json) contains sanitized historical measurements and
source identities from September 9, 2026. It contains no model weights, private
requests, response text, workstation identifiers or machine paths. Porting the
patch does not constitute a new GPU measurement of this checkout.

## Mechanism and dispatch

The original scorer uses four columns of a 16-column BF16 MMA operand for one
query's four heads. The candidate packs four queries from the same request into
those columns, sharing a key-load stream and MMA work. Each query retains its
own position mask. Group masks stay in the MMA layout so the head sum preserves
the original FP32 tree, `(h0 + h1) + (h2 + h3)`.

Dispatch requires BF16 query/cache, exactly 511 rows, four query heads, head
dimension 128, compression ratio four, compressed page size 400 and capacity
65,600. Mixed-request, invalid-request and partial groups use an explicit-row
helper containing the original calculation; the final three rows are partial.
Other direct-call row counts retain the original scorer. Native selection still
chunks an 8,192-row call into sixteen 511-row calls and one 16-row remainder.
The public API, sparse attention, native top-k and expansion stay unchanged.

The inspected main path has 32 static MMA sites for four queries rather than
one. Both arms use two warps, 64 threads and 20,992 shared bytes, with zero
spills and no tensor memory. Registers rise from 78/80 to 100. These instruction
and resource observations do not directly measure DRAM traffic.

## Historical B200 kernel measurements

The table measures the complete 8,192-row scorer plus native top-k and token
expansion, in microseconds per operation. H is one homogeneous request; M
alternates four request IDs, so no complete group takes the packed MMA path.
Mixed-layout gains belong to fallback/program organization in this experiment,
not packed tensor-core utilization.

| History tokens | Layout | Original µs | Candidate µs | Median paired speedup |
| ---: | :---: | ---: | ---: | ---: |
| 8,192 | H | 938.128 | 542.312 | 1.7308× |
| 8,192 | M | 984.496 | 780.456 | 1.2620× |
| 32,768 | H | 2,136.480 | 1,071.040 | 1.9963× |
| 32,768 | M | 1,815.360 | 1,568.552 | 1.1576× |
| 40,527 | H | 2,524.672 | 1,251.000 | 2.0186× |
| 40,527 | M | 2,092.192 | 1,875.328 | 1.1159× |
| 96,516 | H | 5,175.200 | 2,417.432 | 2.1410× |
| 96,516 | M | 3,936.896 | 3,848.080 | 1.0232× |

Each arm uses the same warm synthetic input/cache allocation. Seven paired
groups alternate arm order; each CUDA-event interval contains one graph replay
of four operations with retained outputs. The speedup is the median of paired
ratios, not necessarily the ratio of the displayed medians. All eight cases
won all seven groups. The 79-row scorer and direct 8,192-row scorer are
unchanged controls; their variation is not a patch benefit. The roughly 2.3%
long-history mixed gain is small relative to some unchanged-control variation.

## Correctness and real-worker evidence

The historical sweep passed 676 checks across 38 score cases and 24 timing
cases: 266 exact score/proxy comparisons, including 98 exercising the enabled
dispatch and 467,153,841 defined values across repeated executions. Checks cover
strides, dispatch boundaries, masks, poisoned outputs, graph input/request
changes, stable output addresses, signed cancellation, ties and a nondefault
score divisor. These counts include repeated fixtures and unchanged controls;
they are not independent statistical examples.

All 21 native selection multiset/count/duplicate checks passed. Native top-k
order varies in original repeats; 69 ordered-arm differences remain recorded.
All-zero ties are score/proxy checks, not exact native-set assertions.

The first gather-based implementation failed exact scores because lowering
changed the FP32 reduction tree, despite matching deterministic top-k indices.
The corrected all-row version then regressed a 79-row mixed-request case. The
final patch preserves the MMA-layout reduction and restricts dispatch to 511.

One matched 40,527-token request was profiled in the real NVFP4 worker. Across
six prefill chunks, 948 target scorer launches fell from **83.807495 to
33.758114 ms** in summed GPU duration (**2.482588×**). The grid changed from
`[511,129,1]` to `[128,129,1]`; remainder/decode scorer tuples stayed original.
This is one instrumented request, not 948 independent requests. The candidate
trace includes 624 extra autotuning launches and a 2.634-second QSA CPU scope;
natural output lengths also differ. It is not steady-state request latency.

The full A,C,C,A serving block retained 400 measured attempts over the same
100 historical text-only projects, plus 16 warmups, at concurrency four,
temperature 0.7, medium reasoning, MTP3 and cold prefixes. Ten measured
responses failed the combined response/tool contract across both arms; all
warmups passed. The strict comparator rejected the block without filtering
failures or producing effects/confidence intervals. **No end-to-end serving
gain is accepted, and these failures do not establish kernel causality.**

There is no production soak qualification, broader concurrency/arrival-mix
coverage, multi-GPU evidence, or validation across compiler versions. Observed
request inputs end at 96,516 tokens; this is not full-context-limit coverage.
The original serving kernels remained selected after the campaign.

## Reproduction and checkout validation

Historical runtime: NVIDIA B200/SM100, vLLM `0.1.dev20073+g8e685d198`, PyTorch
`2.13.0+cu130`, CUDA `13.0`, Triton `3.7.1`. The exact container digest,
original/candidate source hashes, harness/helper hashes and per-group timing
samples are in [results.json](results.json).

The historical harness lives in the separate `lovstillery` repository and is
**not included or directly runnable here**. Its saved sweep used rows
`511 79 8192`, histories `8192 32768 40527 96516`, layouts `homogeneous mixed`,
seven groups, four operations per graph, instruction export and a 600-second
bound. Synthetic generator seeds are `9247 + rows + history`. Its geometry is
four query heads, one KV head, dimension 128, compression four and budget 2048;
no model weights or production requests are needed for that microbenchmark.

The backport adds focused CUDA regression cases to the existing test suite:

```bash
.venv/bin/python -m pytest tests/models/qwen4_exp/test_qsa_reference.py \
  -k query_packing -v
```

**This new test command has not been run on a GPU for this backport.** The
historical numerical results above must not be relabeled as a pass of the new
tests or performance validation against a different vLLM revision.
