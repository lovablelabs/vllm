# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.fused_moe.pdl_add import add_shared_and_routed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape", [(1, 2560), (4, 2560), (7, 1000), (64, 2560), (8192, 2560)]
)
def test_add_shared_and_routed_matches_torch(dtype, shape):
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(shape, device="cuda", generator=gen).to(dtype) * 4
    b = torch.randn(shape, device="cuda", generator=gen).to(dtype)
    assert torch.equal(add_shared_and_routed(a, b), a + b)
