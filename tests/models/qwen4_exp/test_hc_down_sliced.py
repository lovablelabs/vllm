# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The sliced HC down gives every row the same bits at every row count: the
clustered kernel at a few rows and bmm + reduce above."""

import os

# The deterministic-serving cuBLASLt setup (split-K off). torch reads it once
# per process, so each test runs in a fresh process that inherits it.
os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

from tests.utils import create_new_process_for_each_test  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_device_capability_family(100):
    pytest.skip("clustered sliced HC down is enabled on SM100", allow_module_level=True)

from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import (  # noqa: E402
    SLICED_MMA_MAX_M,
)
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_down_silu_sliced  # noqa: E402

RANK, HC, K = 320, 4, 10240
N = 336  # rank + hc padded to 16 rows
WIDTH = RANK + HC


@pytest.mark.parametrize("outliers", [False, True])
@create_new_process_for_each_test("spawn")
def test_sliced_hc_down_rows_match_across_row_counts(outliers):
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(64, K, device="cuda", dtype=torch.bfloat16)
    if outliers:
        x[:, ::97] *= 30
    full = _hc_down_silu_sliced(x, w, RANK, HC)[:, :WIDTH]  # bmm + reduce
    for m in range(1, SLICED_MMA_MAX_M + 2):
        rows = _hc_down_silu_sliced(x[:m].contiguous(), w, RANK, HC)[:, :WIDTH]
        assert torch.equal(rows, full[:m]), m
