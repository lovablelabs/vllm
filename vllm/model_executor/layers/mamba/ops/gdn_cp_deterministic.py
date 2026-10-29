# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN prefill on FlashInfer's chunk-parallel kernels with deterministic outputs.

FlashInfer's chunk-parallel (CP) GDN prefill on SM100 runs in four steps: T
tiles per 64-token chunk, a transfer matrix M and a local state N per CP
segment (from a zero state), a sequential fixup over the segments, and the
outputs of every segment from its incoming state. With the segment length set
to the KV block size and the fixup below in place of FlashInfer's, a sequence's
outputs depend only on its own tokens and initial state: the segment-end states
are rounded to bf16, as the bf16 state cache rounds them between one-block
prefill chunks, so a prefill call over several blocks gives the same bits as one
call per block, and as a call that resumes from a cached block's state.
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_D = 128


def _cp_ops():
    from flashinfer.gdn_kernels.blackwell.gdn_cp_prefill import (
        cp_delta_rule_mn_precompute_dsl_sm100,
        cp_delta_rule_prefill_dsl_sm100,
        cp_delta_rule_t_precompute_dsl_sm100,
    )

    return (
        cp_delta_rule_t_precompute_dsl_sm100,
        cp_delta_rule_mn_precompute_dsl_sm100,
        cp_delta_rule_prefill_dsl_sm100,
    )


def deterministic_cp_unavailable_reason(head_k_dim: int, head_v_dim: int) -> str | None:
    """Why the CP path cannot run here, or None."""
    if not torch.cuda.is_available():
        return "needs a CUDA device"
    if torch.cuda.get_device_capability()[0] != 10:
        return "needs an SM100 GPU"
    if head_k_dim != _D or head_v_dim != _D:
        return f"needs 128-dim heads (got k={head_k_dim}, v={head_v_dim})"
    try:
        _cp_ops()
    except Exception as e:  # noqa: BLE001 - private FlashInfer API
        return f"is unavailable in this FlashInfer ({e})"
    return None


@triton.jit
def _deterministic_cp_fixup_kernel(
    transfer,
    local_state,
    initial_state,
    fixed_state,
    final_state,
    cu_seqlens,
    SEGMENT: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    # One program per (sequence, head, row tile) of the [D, D] state: row r of
    # S @ M + N only needs row r of S, so the programs are independent.
    i_n, i_h, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(cu_seqlens + i_n).to(tl.int64)
    end = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    # FlashInfer's CP workspace index of the sequence's first segment.
    m = tl.minimum(start, i_n.to(tl.int64))
    first = m + (start - m) // SEGMENT
    num_segments = (end - start + SEGMENT - 1) // SEGMENT

    rows = i_v * BV + tl.arange(0, BV)
    cols = tl.arange(0, D)
    o_k = tl.arange(0, BK)
    tile = rows[:, None] * D + cols[None, :]
    prev = initial_state + (i_n.to(tl.int64) * H + i_h) * D * D
    for j in range(num_segments):
        base = ((first + j) * H + i_h) * D * D
        acc = tl.load(local_state + base + tile)
        for k0 in range(0, D, BK):
            b_s = tl.load(prev + rows[:, None] * D + (k0 + o_k)[None, :])
            b_m = tl.load(transfer + base + (k0 + o_k)[:, None] * D + cols[None, :])
            acc = tl.dot(b_s, b_m, acc, input_precision="ieee")
        # The state the bf16 cache would hold at this block boundary.
        acc = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        tl.store(fixed_state + base + tile, acc)
        # The next segment reads these rows back (written by other threads).
        tl.debug_barrier()
        prev = fixed_state + base
    tl.store(
        final_state + (i_n.to(tl.int64) * H + i_h) * D * D + tile,
        tl.load(prev + tile),
    )


def deterministic_cp_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    segment: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GDN prefill with CP segments of ``segment`` tokens and bf16 segment states.

    q, k: [1, T, Hk, 128] (l2-normalized); v: [1, T, H, 128]; g: [1, T, H] log
    gates; beta: [1, T, H]; initial_state: [N, H, 128, 128] (FLA layout, the
    rows' cached states, zeros for new rows); cu_seqlens: [N + 1]. ``segment``
    is the KV block size; every sequence must start at a block boundary.
    Returns the output [1, T, H, 128] and the final states [N, H, 128, 128]
    (fp32 holding bf16 values).
    """
    t_precompute, mn_precompute, prefill = _cp_ops()
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()
    alpha = torch.exp(g.squeeze(0).float()).contiguous()
    beta = beta.squeeze(0).float().contiguous()
    total, num_heads, d = v.shape
    num_seqs = cu_seqlens.numel() - 1
    h0 = initial_state.float().contiguous()

    t = t_precompute(k, beta, cu_seqlens, total, max_seqlen=total)
    transfer, local_state = mn_precompute(
        k, v, t, alpha, cu_seqlens, total, cp_chunk_len=segment, max_seqlen=total
    )
    fixed_state = torch.empty_like(transfer)
    final_state = torch.empty_like(h0)
    bv, bk = 16, 32
    _deterministic_cp_fixup_kernel[(num_seqs, num_heads, d // bv)](
        transfer,
        local_state,
        h0,
        fixed_state,
        final_state,
        cu_seqlens,
        SEGMENT=segment,
        H=num_heads,
        D=d,
        BV=bv,
        BK=bk,
        num_warps=4,
    )
    o = torch.empty_like(v)
    prefill(
        o,
        None,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        d**-0.5,
        cu_seqlens,
        total,
        cp_chunk_len=segment,
        max_seqlen=total,
        initial_state=h0,
    )
    return o.unsqueeze(0), final_state
