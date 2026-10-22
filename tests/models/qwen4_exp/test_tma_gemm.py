# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen4Exp TMA GEMM returns cuBLASLt's bits (split-K off) at every row
count, so deterministic serving can use it at a few rows without changing
outputs."""

import os

# The deterministic-serving cuBLASLt setup (split-K off). torch reads it once
# per process, so each test runs in a fresh process that inherits it.
os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "1")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tests.utils import create_new_process_for_each_test  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_device_capability_family(100):
    pytest.skip("TMA GEMM plans are tuned for SM100", allow_module_level=True)

from vllm.models.qwen4_exp.nvidia.low_latency_gemm import (  # noqa: E402
    QWEN4_EXP_SM100_TMA_GEMM_PLANS,
    _qwen4_exp_tma_gemm,
)
from vllm.models.qwen4_exp.nvidia.ops.tma_gemm import (  # noqa: E402
    TmaGemmConfig,
    tma_gemm,
)

ROWS = [1, 2, 3, 4, 5, 8, 10, 16, 31, 32, 33, 64, 65, 256, 1776]


@pytest.mark.parametrize("shape", sorted(QWEN4_EXP_SM100_TMA_GEMM_PLANS))
@create_new_process_for_each_test("spawn")
def test_tma_gemm_matches_cublas(shape):
    torch.manual_seed(0)
    n, k = shape
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(max(ROWS), k, device="cuda", dtype=torch.bfloat16)
    x[:, ::97] *= 30
    config = QWEN4_EXP_SM100_TMA_GEMM_PLANS[shape]
    wide = TmaGemmConfig(
        4096, config.block_n, config.block_k, config.num_stages, config.l2_prefetch
    )
    # The dispatching op gives every row the same bits at every row count,
    # whichever kernel serves the call.
    full = _qwen4_exp_tma_gemm(x, w)
    for m in ROWS:
        rows = x[:m].contiguous()
        assert torch.equal(_qwen4_exp_tma_gemm(rows, w), full[:m]), m
        # The planned config returns cuBLASLt's bits. cuBLASLt itself is not
        # row-count invariant for a single output column on some inputs, so
        # that shape is compared at a few rows only.
        if n > 1 or m <= 16:
            ref = F.linear(rows, w)
            assert torch.equal(tma_gemm(rows, w, wide, launch_pdl=True), ref), m


@create_new_process_for_each_test("spawn")
def test_tma_gemm_strided_rows_and_graph():
    torch.manual_seed(0)
    n, k = 640, 2560
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    # Rows of a wider buffer (16-byte aligned row stride) use TMA; an odd row
    # stride falls back to cuBLASLt. Both give cuBLASLt's bits.
    wide = torch.randn(8, k + 64, device="cuda", dtype=torch.bfloat16)
    odd = torch.randn(8, k + 3, device="cuda", dtype=torch.bfloat16)
    for x in (wide[:, :k], odd[:, :k]):
        assert torch.equal(_qwen4_exp_tma_gemm(x, w), F.linear(x, w))

    x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x, w)
    _qwen4_exp_tma_gemm(x, w)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = _qwen4_exp_tma_gemm(x, w)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, ref)
