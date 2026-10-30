# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA sparse decode attention whose bits do not depend on the batch size.

Deterministic serving needs a decode row's attention to give the same bits at
every batch size, while few-row batches need many programs per row to fill the
GPU and large batches need few. Split-K with the usual online softmax ties the
bits to the split count. Here the row's selected columns are cut into leaves
of 128 (16 for a 2048-token budget, plus a tail leaf for the up to three extra
columns the compressed-block expansion adds); each leaf gets its own softmax
partial (its max ``m``, row sum ``l`` and ``A = P.V`` from zero), and the
partials fold in one fixed tree: four left folds of four leaves, a left fold
of those four nodes, then the tail leaf. Every fold is the same operation
(``m = max``, ``l`` and ``A`` rescaled by ``exp2`` and fused multiply-adds),
so a program may compute any aligned subtree: one leaf (16 programs per row)
or four (4 programs per row), the tail leaf its own program, and the merge
kernel folds the rest. Leaves past the selection count are the fold's exact
identity (``m = -1e20``, ``l = 0``, ``A = 0``) and are skipped.

The row sums go through the tensor cores (a dot with ones), so the warp count
leaves the bits unchanged too. Prefill rows keep the split kernel.
"""

import torch

from vllm.triton_utils import tl, triton

LEAF = 128
NUM_LEAVES = 16
NEG = tl.constexpr(-1.0e20)


@triton.jit
def _fold_scale(m1, l1, m2, l2):
    m = tl.maximum(m1, m2)
    a1 = tl.math.exp2(m1 - m)
    a2 = tl.math.exp2(m2 - m)
    return m, tl.fma(l1, a1, l2 * a2), a1, a2


@triton.jit
def _fold_acc(acc1, a1, acc2, a2):
    # a1, a2: per-row scales broadcastable to the accumulators.
    return tl.fma(
        acc1,
        tl.broadcast_to(a1, acc1.shape),
        acc2 * tl.broadcast_to(a2, acc2.shape),
    )


@triton.jit
def _fold_leaf(
    m1,
    l1,
    x0,
    x1,
    x2,
    x3,
    query,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    row,
    request,
    safe_request,
    kv_head,
    leaf,
    score_scale,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """Fold leaf ``leaf``'s partial into the running node (m1, l1, x0..x3)."""
    SLICE: tl.constexpr = HEAD_DIM // 4
    column_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, HEAD_DIM)
    slice_offsets = tl.arange(0, SLICE)
    columns = leaf * BLOCK_N + column_offsets
    logical_token = tl.load(
        indices_ptr + row * stride_indices_row + columns,
        mask=columns < TOPK,
        other=-1,
    )
    safe_token = tl.maximum(logical_token, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token % PAGE_SIZE
    valid = (
        (request >= 0)
        & (request < num_requests)
        & (logical_token >= 0)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        block_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    # physical_page * block stride can overflow int32 for large caches.
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)
    keys = tl.load(
        k_cache_ptr
        + safe_page[None, :] * stride_k_block
        + page_offset[None, :] * stride_k_token
        + kv_head * stride_k_head
        + dim_offsets[:, None],
        mask=valid[None, :],
        other=0.0,
    )
    if IS_FP8:
        keys = keys.to(query.dtype)
    scores = tl.dot(query, keys) * score_scale
    scores = tl.where(valid[None, :], scores, NEG)
    m2 = tl.max(scores, axis=1)
    probabilities = tl.where(valid[None, :], tl.math.exp2(scores - m2[:, None]), 0.0)
    if IS_FP8:
        # fp8 V dequantizes to fp16 for the P.V dot (P <= 1).
        p_values = probabilities.to(tl.float16)
    else:
        p_values = probabilities.to(tl.bfloat16)
    # Row sums on the tensor cores (a fixed K order whatever the warp count).
    l2 = tl.max(tl.dot(p_values, tl.full((BLOCK_N, 16), 1.0, p_values.dtype)), axis=1)
    m, l, a1, a2 = _fold_scale(m1, l1, m2, l2)
    a1 = a1[:, None]
    a2 = a2[:, None]
    # The leaf's P.V in four head-dim slices, each folded in right away, so a
    # leaf holds a quarter of a fresh accumulator at a time.
    v_base = (
        v_cache_ptr
        + safe_page[:, None] * stride_v_block
        + page_offset[:, None] * stride_v_token
        + kv_head * stride_v_head
        + slice_offsets[None, :]
    )
    v = tl.load(v_base, mask=valid[:, None], other=0.0).to(p_values.dtype)
    x0 = _fold_acc(x0, a1, tl.dot(p_values, v), a2)
    v = tl.load(v_base + SLICE, mask=valid[:, None], other=0.0).to(p_values.dtype)
    x1 = _fold_acc(x1, a1, tl.dot(p_values, v), a2)
    v = tl.load(v_base + 2 * SLICE, mask=valid[:, None], other=0.0).to(p_values.dtype)
    x2 = _fold_acc(x2, a1, tl.dot(p_values, v), a2)
    v = tl.load(v_base + 3 * SLICE, mask=valid[:, None], other=0.0).to(p_values.dtype)
    x3 = _fold_acc(x3, a1, tl.dot(p_values, v), a2)
    return m, l, x0, x1, x2, x3


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_tree_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    node_m_ptr,
    node_l_ptr,
    node_acc_ptr,
    softmax_scale,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    LEAVES_PER_PROGRAM: tl.constexpr,
    MAIN_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """One leaf or one level-1 node (four leaves) per program, stored for the
    merge kernel; program MAIN_PROGRAMS (when the selection has a tail) takes
    the tail leaf."""
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    program = tl.program_id(2)
    SLICE: tl.constexpr = HEAD_DIM // 4
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    # The packed selection buffer's trailing column is the row's valid count.
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    leaf_end = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )
    # log2 units for exp2; fp8 caches fold the K dequant scale in on the host.
    score_scale = softmax_scale * 1.4426950408889634
    m = tl.full((BLOCK_M,), NEG, tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    x0 = tl.zeros((BLOCK_M, SLICE), tl.float32)
    x1 = tl.zeros((BLOCK_M, SLICE), tl.float32)
    x2 = tl.zeros((BLOCK_M, SLICE), tl.float32)
    x3 = tl.zeros((BLOCK_M, SLICE), tl.float32)
    is_main = program < MAIN_PROGRAMS
    start = tl.where(is_main, program * LEAVES_PER_PROGRAM, MAIN_PROGRAMS * LEAVES_PER_PROGRAM)
    count = tl.where(is_main, LEAVES_PER_PROGRAM, 1)
    for leaf in range(start, tl.minimum(start + count, leaf_end)):
        m, l, x0, x1, x2, x3 = _fold_leaf(
            m,
            l,
            x0,
            x1,
            x2,
            x3,
            query,
            k_cache_ptr,
            v_cache_ptr,
            indices_ptr,
            block_table_ptr,
            row,
            request,
            safe_request,
            kv_head,
            leaf,
            score_scale,
            stride_k_block,
            stride_k_token,
            stride_k_head,
            stride_v_block,
            stride_v_token,
            stride_v_head,
            stride_indices_row,
            stride_table_req,
            num_cache_blocks,
            num_requests,
            TOPK,
            PAGE_SIZE,
            PAGE_TABLE_WIDTH,
            HEAD_DIM,
            BLOCK_N,
            IS_FP8,
        )
    head_mask = head_offsets < GROUP_SIZE
    heads = (program * num_rows + row).to(tl.int64) * NUM_QUERY_HEADS
    heads += first_head + head_offsets
    tl.store(node_m_ptr + heads, m, mask=head_mask)
    tl.store(node_l_ptr + heads, l, mask=head_mask)
    acc = node_acc_ptr + heads[:, None] * HEAD_DIM + tl.arange(0, SLICE)[None, :]
    tl.store(acc, x0, mask=head_mask[:, None])
    tl.store(acc + SLICE, x1, mask=head_mask[:, None])
    tl.store(acc + 2 * SLICE, x2, mask=head_mask[:, None])
    tl.store(acc + 3 * SLICE, x3, mask=head_mask[:, None])


@triton.jit
def _split4(x):
    """The four entries of x's last dim (size 4), in order."""
    lo0, lo1 = tl.split(tl.reshape(x, x.shape[:-1] + [2, 2]))
    t0, t2 = tl.split(lo0)
    t1, t3 = tl.split(lo1)
    return t0, t1, t2, t3


@triton.jit
def _fold_rows(m1, l1, x1, m2, l2, x2):
    m, l, a1, a2 = _fold_scale(m1, l1, m2, l2)
    return m, l, _fold_acc(x1, a1[:, None], x2, a2[:, None])


@triton.jit
def _fold_vector(m1, l1, x1, m2, l2, x2):
    m, l, a1, a2 = _fold_scale(m1, l1, m2, l2)
    return m, l, _fold_acc(x1, a1, x2, a2)


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_tree_merge_kernel(
    node_m_ptr,
    node_l_ptr,
    node_acc_ptr,
    output_ptr,
    output_gate_ptr,
    output_scale,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_NODES: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    """Fold one (row, query head)'s stored leaves (16) or level-1 nodes (4) up
    the tree, then the tail leaf (node NUM_NODES), normalize, round to bf16 and
    apply the sigmoid output gate. All nodes load up front; each level folds
    across its groups at once (the same per-element operations in the same
    order as one leaf at a time)."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, HEAD_DIM)
    nodes = tl.arange(0, NUM_NODES)
    idx = (nodes.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head
    m_all = tl.load(node_m_ptr + idx)
    l_all = tl.load(node_l_ptr + idx)
    x_all = tl.load(node_acc_ptr + idx[:, None] * HEAD_DIM + dims[None, :])
    if NUM_NODES == 16:
        # [group, leaf] -> level-1 nodes [group].
        m0, m1, m2, m3 = _split4(tl.reshape(m_all, [4, 4]))
        l0, l1, l2, l3 = _split4(tl.reshape(l_all, [4, 4]))
        x0, x1, x2, x3 = _split4(
            tl.permute(tl.reshape(x_all, [4, 4, HEAD_DIM]), (0, 2, 1))
        )
        mg = tl.full((4,), NEG, tl.float32)
        lg = tl.zeros((4,), tl.float32)
        xg = tl.zeros((4, HEAD_DIM), tl.float32)
        mg, lg, xg = _fold_rows(mg, lg, xg, m0, l0, x0)
        mg, lg, xg = _fold_rows(mg, lg, xg, m1, l1, x1)
        mg, lg, xg = _fold_rows(mg, lg, xg, m2, l2, x2)
        mg, lg, xg = _fold_rows(mg, lg, xg, m3, l3, x3)
    else:
        mg = m_all
        lg = l_all
        xg = x_all
    # Level-1 nodes -> root.
    m0, m1, m2, m3 = _split4(mg)
    l0, l1, l2, l3 = _split4(lg)
    x0, x1, x2, x3 = _split4(tl.permute(xg, (1, 0)))
    mr = tl.full((), NEG, tl.float32)
    lr = tl.zeros((), tl.float32)
    xr = tl.zeros((HEAD_DIM,), tl.float32)
    mr, lr, xr = _fold_vector(mr, lr, xr, m0, l0, x0)
    mr, lr, xr = _fold_vector(mr, lr, xr, m1, l1, x1)
    mr, lr, xr = _fold_vector(mr, lr, xr, m2, l2, x2)
    mr, lr, xr = _fold_vector(mr, lr, xr, m3, l3, x3)
    if HAS_TAIL:
        tail = (NUM_NODES * num_rows + row).to(tl.int64) * NUM_QUERY_HEADS + head
        mt = tl.load(node_m_ptr + tail)
        lt = tl.load(node_l_ptr + tail)
        xt = tl.load(node_acc_ptr + tail * HEAD_DIM + dims)
        mr, lr, xr = _fold_vector(mr, lr, xr, mt, lt, xt)
    # The fp8 V dequant scale (1.0 for bf16) goes into the normalizer.
    out = tl.where(lr > 0, xr * (output_scale / tl.maximum(lr, 1.0e-20)), 0.0)
    # The unfused path's bf16 attention-output rounding before the fp32 gate.
    out = out.to(output_ptr.dtype.element_ty).to(tl.float32)
    gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dims
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dims,
        out * tl.sigmoid(gate),
    )


def qsa_tree_supported(selection_width: int, head_dim: int) -> bool:
    """A 2048-token budget's selection: 16 leaves and at most a tail leaf."""
    return (
        LEAF * NUM_LEAVES <= selection_width <= LEAF * (NUM_LEAVES + 1)
        and head_dim % 64 == 0
    )


def qsa_tree_layout(num_rows: int, num_kv_heads: int) -> tuple[int, int]:
    """(programs per row, warps): one leaf per program at few rows, four at
    many (B200, Qwen4Exp TP2: 12 query heads on 1 KV head, 2051 columns)."""
    if num_rows * num_kv_heads <= 40:
        return NUM_LEAVES, 4
    return NUM_LEAVES // 4, 4


def qsa_tree_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    output_gate: torch.Tensor,
    softmax_scale: float,
    output_scale: float,
) -> torch.Tensor:
    rows, num_query_heads, head_dim = q.shape
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    programs, warps = qsa_tree_layout(rows, num_kv_heads)
    has_tail = logical_indices.shape[1] - 1 > LEAF * NUM_LEAVES
    nodes = programs + int(has_tail)
    node_m = torch.empty(
        (nodes, rows, num_query_heads), dtype=torch.float32, device=q.device
    )
    node_l = torch.empty_like(node_m)
    node_acc = torch.empty(
        (nodes, rows, num_query_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    _qsa_tree_kernel[(rows, num_kv_heads, nodes)](
        q,
        key_cache,
        value_cache,
        logical_indices,
        block_table,
        token_to_req,
        node_m,
        node_l,
        node_acc,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        rows,
        key_cache.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1] - 1,
        PAGE_SIZE=key_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=num_query_heads,
        LEAVES_PER_PROGRAM=NUM_LEAVES // programs,
        MAIN_PROGRAMS=programs,
        BLOCK_M=triton.next_power_of_2(group_size),
        BLOCK_N=LEAF,
        IS_FP8=key_cache.dtype == torch.float8_e4m3fn,
        num_warps=warps,
        num_stages=2,
    )
    _qsa_tree_merge_kernel[(rows, num_query_heads)](
        node_m,
        node_l,
        node_acc,
        out,
        output_gate,
        output_scale,
        out.stride(0),
        out.stride(1),
        output_gate.stride(0),
        output_gate.stride(1),
        rows,
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS=num_query_heads,
        NUM_NODES=programs,
        HAS_TAIL=has_tail,
        num_warps=4 if programs == NUM_LEAVES else 2,
        num_stages=1,
    )
    return out
