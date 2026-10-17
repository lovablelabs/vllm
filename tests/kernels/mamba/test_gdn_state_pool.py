# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer's SM100 GDN state-pool mode must match the gather /
zero / fp32 / scatter path bit for bit (outputs and the SSM cache), with new and
continuing sequences mixed, a padded first-dimension stride and strided index /
mask views."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="FlashInfer's GDN state-pool mode is SM100-only",
)
H, HV, K, V = 8, 24, 128, 128


def _l2(x):
    norm = x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x.float() / norm).to(x.dtype)


@pytest.mark.parametrize(
    "lens, has_init",
    [
        ([1500], [True]),
        ([300, 1200, 700], [True, False, True]),
        ([64, 900], [False, False]),
    ],
)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("strided_metadata", [False, True])
def test_pool_mode_matches_gather_path(lens, has_init, padded, strided_metadata):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
    )
    from vllm.model_executor.layers.mamba.ops.gdn_state_pool import (
        fi_chunk_gated_delta_rule_pool,
        pool_mode_supported,
    )

    torch.manual_seed(0)
    dev = "cuda"
    T = sum(lens)
    q = _l2(torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16))
    k = _l2(torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16))
    v = torch.randn(1, T, HV, V, device=dev, dtype=torch.bfloat16)
    g = -F.softplus(torch.randn(1, T, HV, device=dev) * 2 + 2)
    beta = torch.sigmoid(torch.randn(1, T, HV, device=dev))
    cu = torch.tensor(
        [0] + torch.tensor(lens).cumsum(0).tolist(), device=dev, dtype=torch.int32
    )
    rows = 32
    if padded:  # rows of a larger page (the cache packs other state beside it)
        page = torch.randn(rows, HV * V * K + 4096, device=dev).to(torch.bfloat16)
        pool = page[:, : HV * V * K].view(rows, HV, V, K)
    else:
        pool = (torch.randn(rows, HV, V, K, device=dev) * 0.5).to(torch.bfloat16)
    assert pool_mode_supported(pool)
    idx = (torch.randperm(rows - 1, device=dev)[: len(lens)] + 1).to(torch.int32)
    mask = torch.tensor(has_init, device=dev)

    pool_a = pool.clone()
    init = pool_a[idx]
    init[~mask, ...] = 0
    o_a, h_a = fi_chunk_gated_delta_rule(
        q, k, v, g, beta, init, True, cu_seqlens=cu, use_qk_l2norm_in_kernel=False
    )
    pool_a[idx] = h_a.to(pool_a.dtype)

    if padded:  # keep the padded stride for the pool-mode side
        pool_b = page.clone()[:, : HV * V * K].view(rows, HV, V, K)
    else:
        pool_b = pool.clone()
    if strided_metadata:  # the metadata may hand over column views
        idx = torch.stack([idx, torch.zeros_like(idx)], dim=1)[:, 0]
        mask = torch.stack([mask, ~mask], dim=1)[:, 0]
    o_b = fi_chunk_gated_delta_rule_pool(
        q, k, v, g, beta, pool_b, idx, mask, cu, use_qk_l2norm_in_kernel=False
    )
    assert torch.equal(o_a, o_b)
    assert torch.equal(pool_a, pool_b)


def test_pool_mode_rejects_a_transposed_layout():
    from vllm.model_executor.layers.mamba.ops.gdn_state_pool import (
        pool_mode_supported,
    )

    pool = torch.zeros(4, HV, K, V, dtype=torch.bfloat16, device="cuda")
    pool = pool.transpose(-1, -2)
    assert not pool_mode_supported(pool)
