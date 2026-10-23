# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""With the warp count pinned (deterministic serving), no autotuning choice of
the FLA GDN prefill kernels changes the output bits."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FLA GDN kernels need CUDA", allow_module_level=True)

from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    chunk,
    chunk_delta_h,
    chunk_o,
    chunk_scaled_dot_kkt,
    cumsum,
    solve_tril,
    wy_fast,
)
from vllm.triton_utils import triton  # noqa: E402

H, HK, DK, DV = 16, 8, 128, 128  # value heads, key heads (GQA), head dims


def _autotuners():
    tuners = []
    for module in (
        chunk_delta_h,
        chunk_o,
        chunk_scaled_dot_kkt,
        cumsum,
        solve_tril,
        wy_fast,
    ):
        for kernel in vars(module).values():
            while kernel is not None and not isinstance(
                kernel, triton.runtime.autotuner.Autotuner
            ):
                kernel = getattr(kernel, "fn", None)
            if kernel is not None and kernel not in tuners:
                tuners.append(kernel)
    return tuners


def _run(lengths, beta_dtype):
    torch.manual_seed(0)
    T = sum(lengths)
    q = F.normalize(torch.randn(1, T, HK, DK, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, T, HK, DK, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(1, T, H, DV, device="cuda").bfloat16()
    g = F.logsigmoid(torch.randn(1, T, H, device="cuda") + 4)
    # Serving passes an fp32 beta, which makes the k k^T dot fp32.
    beta = torch.rand(1, T, H, device="cuda").to(beta_dtype)
    h0 = (torch.randn(len(lengths), H, DV, DK, device="cuda") * 0.05).bfloat16()
    cu_seqlens = torch.tensor([0, *lengths], device="cuda").cumsum(0).to(torch.int32)
    return chunk.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16])
def test_pinned_autotuning_keeps_bits(beta_dtype):
    lengths = [1792, 900]
    chunk.pin_autotune_num_warps()
    tuners = _autotuners()
    saved = {tuner: list(tuner.configs) for tuner in tuners}
    try:
        _run(lengths, beta_dtype)
        used = [tuner for tuner in tuners if tuner.cache]
        assert used
        for tuner in used:
            tuner.configs = saved[tuner][:1]
            tuner.cache.clear()
        ref_o, ref_h = _run(lengths, beta_dtype)
        for tuner in used:
            for config in saved[tuner]:
                tuner.configs = [config]
                tuner.cache.clear()
                o, h = _run(lengths, beta_dtype)
                name = tuner.base_fn.__name__
                assert torch.equal(o, ref_o), f"{name} {config}"
                assert torch.equal(h, ref_h), f"{name} {config}"
            tuner.configs = saved[tuner][:1]
            tuner.cache.clear()
    finally:
        for tuner, configs in saved.items():
            tuner.configs = configs
            tuner.cache.clear()
