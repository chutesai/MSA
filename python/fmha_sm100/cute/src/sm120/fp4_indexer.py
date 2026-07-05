# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""SM120 Triton FP4 indexer: block max-scores via native block-scaled MMA.

Implements the ``fp4_indexer_block_scores`` contract (MXFP4, public scale
layout) with ``tl.dot_scaled`` e2m1 x e2m1, which lowers to sm_120's native
block-scaled tensor-core MMA. Block scoring is a dense, compute-bound pass
whose output only ranks KV pages for top-k selection, which is what makes
FP4 the right precision here.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_PAGE_SIZE = 128
_PACKED_BYTES = 64  # D=128 packed two FP4 values per byte
_MX_SCALE_GROUPS = 4  # D=128 / 32-element ue8m0 groups
_BLOCK_Q = 128

# Kernel-visible constexpr twins of the host constants above.
_PAGE_C = tl.constexpr(_PAGE_SIZE)
_PB_C = tl.constexpr(_PACKED_BYTES)
_G_C = tl.constexpr(_MX_SCALE_GROUPS)


@triton.jit
def _fp4_indexer_scores_kernel(
    q_fp4,
    k_fp4,
    q_scale,
    k_scale,
    cu_q,
    cu_k,
    cu_pages,
    kv_indices,
    qo_offset,
    scores,
    total_q: tl.constexpr,
    heads_q: tl.constexpr,
    heads_k: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    max_q_tiles: tl.constexpr,
    max_k_tiles: tl.constexpr,
    causal: tl.constexpr,
    has_qo_offset: tl.constexpr,
    BQ: tl.constexpr,
):
    pid = tl.program_id(0)
    hq = tl.program_id(1)
    b = pid // max_q_tiles
    qt = pid % max_q_tiles
    hk = hq // qhead_per_kv

    q_begin = tl.load(cu_q + b)
    q_len = tl.load(cu_q + b + 1) - q_begin
    if qt * BQ >= q_len:
        return
    k_len = tl.load(cu_k + b + 1) - tl.load(cu_k + b)
    page_cursor = tl.load(cu_pages + b)
    pages_b = (k_len + _PAGE_C - 1) // _PAGE_C

    offs_q = qt * BQ + tl.arange(0, BQ)  # batch-local query index
    q_valid = offs_q < q_len
    q_abs = q_begin + offs_q
    offs_b64 = tl.arange(0, _PB_C)
    offs_g = tl.arange(0, _G_C)
    offs_n = tl.arange(0, _PAGE_C)

    q_ptrs = q_fp4 + (q_abs[:, None] * heads_q + hq) * _PB_C + offs_b64[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None], other=0)
    qs_ptrs = q_scale + (q_abs[:, None] * heads_q + hq) * _G_C + offs_g[None, :]
    q_sc = tl.load(qs_ptrs, mask=q_valid[:, None], other=0)

    if has_qo_offset:
        offset = tl.load(qo_offset + b)
    else:
        offset = k_len - q_len

    kt_hi = pages_b
    if causal:
        # Last visible KV position for the tile's last valid query row.
        q_tile_last = tl.minimum(qt * BQ + BQ - 1, q_len - 1)
        visible_limit = q_tile_last + offset
        if visible_limit < 0:
            return
        kt_hi = tl.minimum(pages_b, visible_limit // _PAGE_C + 1)

    for ktile in range(0, kt_hi):
        physical_page = tl.load(kv_indices + page_cursor + ktile)
        k_start = ktile * _PAGE_C
        tok_valid = k_start + offs_n < k_len
        k_ptrs = (
            k_fp4
            + ((physical_page * heads_k + hk) * _PAGE_C + offs_n[None, :]) * _PB_C
            + offs_b64[:, None]
        )
        k_tile = tl.load(k_ptrs, mask=tok_valid[None, :], other=0)
        ks_ptrs = (
            k_scale
            + ((physical_page * heads_k + hk) * _PAGE_C + offs_n[:, None]) * _G_C
            + offs_g[None, :]
        )
        k_sc = tl.load(ks_ptrs, mask=tok_valid[:, None], other=0)

        logits = tl.dot_scaled(
            q_tile, q_sc, "e2m1", k_tile, k_sc, "e2m1",
            lhs_k_pack=True, rhs_k_pack=True, out_dtype=tl.float32,
        )
        visible = q_valid[:, None] & tok_valid[None, :]
        if causal:
            visible = visible & (offs_q[:, None] + offset >= (k_start + offs_n)[None, :])
        logits = tl.where(visible, logits, -float("inf"))
        page_max = tl.max(logits, axis=1)
        row_has_visible = tl.max(visible.to(tl.int32), axis=1) > 0
        tl.store(
            scores + (hq * max_k_tiles + ktile) * total_q + q_abs,
            page_max,
            mask=q_valid & row_has_visible,
        )


def fp4_indexer_block_scores_triton(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_page_offsets: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    kv_indices: torch.Tensor,
    fp4_format: str,
    causal: bool = False,
    qo_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SM120 implementation of the ``fp4_indexer_block_scores`` contract.

    MXFP4 with the public scale layout only: q_scale ``[total_q, Hq, 4]`` and
    k_scale ``[pages, Hk, 128, 4]`` ue8m0 bytes. Returns
    ``[Hq, ceil(max_seqlen_k / 128), total_q]`` float32 with ``-inf`` outside
    the valid / visible range.
    """
    if str(fp4_format).lower() != "mxfp4":
        raise NotImplementedError(
            "SM120 FP4 indexer currently supports fp4_format='mxfp4' "
            f"(ue8m0 scales); got {fp4_format!r}"
        )
    q_bytes = q_fp4.view(torch.uint8)
    k_bytes = k_fp4.view(torch.uint8)
    total_q, heads_q, packed = (int(v) for v in q_bytes.shape)
    page_count, heads_k, page_size, k_packed = (int(v) for v in k_bytes.shape)
    if packed != _PACKED_BYTES or k_packed != _PACKED_BYTES or page_size != _PAGE_SIZE:
        raise ValueError("FP4 indexer expects D=128 packed as 64 bytes and 128-token pages")
    if heads_q % heads_k != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    q_sc = q_scale.view(torch.uint8)
    k_sc = k_scale.view(torch.uint8)
    if tuple(q_sc.shape) != (total_q, heads_q, _MX_SCALE_GROUPS):
        raise ValueError("q_scale must have public layout [total_q, Hq, 4] for mxfp4")
    if tuple(k_sc.shape) != (page_count, heads_k, _PAGE_SIZE, _MX_SCALE_GROUPS):
        raise ValueError("k_scale must have public layout [pages, Hk, 128, 4] for mxfp4")
    batch = int(cu_seqlens_q.shape[0]) - 1
    max_k_tiles = (int(max_seqlen_k) + _PAGE_SIZE - 1) // _PAGE_SIZE
    if max_k_tiles == 0 or batch <= 0:
        return torch.full((heads_q, 0, total_q), float("-inf"), dtype=torch.float32, device=q_fp4.device)
    max_q_tiles = (int(max_seqlen_q) + _BLOCK_Q - 1) // _BLOCK_Q
    scores = torch.full(
        (heads_q, max_k_tiles, total_q), float("-inf"), dtype=torch.float32, device=q_fp4.device
    )
    if qo_offset is None:
        qo_offset_arg = cu_seqlens_q
        has_qo_offset = False
    else:
        qo_offset_arg = qo_offset
        has_qo_offset = True
    grid = (batch * max_q_tiles, heads_q)
    _fp4_indexer_scores_kernel[grid](
        q_bytes.contiguous(),
        k_bytes.contiguous(),
        q_sc.contiguous(),
        k_sc.contiguous(),
        cu_seqlens_q,
        cu_seqlens_k,
        cu_page_offsets,
        kv_indices,
        qo_offset_arg,
        scores,
        int(total_q),
        int(heads_q),
        int(heads_k),
        int(heads_q // heads_k),
        int(max_q_tiles),
        int(max_k_tiles),
        bool(causal),
        bool(has_qo_offset),
        BQ=_BLOCK_Q,
        num_warps=8,
        num_stages=2,
    )
    return scores


@triton.jit
def _fp4_indexer_topk_kernel(
    q_fp4,
    k_fp4,
    q_scale,
    k_scale,
    cu_q,
    cu_k,
    cu_pages,
    kv_indices,
    qo_offset,
    q2k,
    total_q: tl.constexpr,
    heads_q: tl.constexpr,
    heads_k: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    max_q_tiles: tl.constexpr,
    causal: tl.constexpr,
    has_qo_offset: tl.constexpr,
    TOPK: tl.constexpr,
    FORCE_DIAG: tl.constexpr,
    FORCE_BEGIN: tl.constexpr,
    BQ: tl.constexpr,
):
    """Fused block scoring + per-query top-k selection.

    Identical page loop to the scores kernel, but scores never leave the
    program: a running (score, page) insertion list per query row lives in
    registers, forced slots are pre-seeded with +inf scores, and the epilogue
    sorts each row's page ids ascending with a -1 tail — emitting the q2k
    selection contract directly.
    """
    pid = tl.program_id(0)
    hq = tl.program_id(1)
    b = pid // max_q_tiles
    qt = pid % max_q_tiles
    hk = hq // qhead_per_kv

    q_begin = tl.load(cu_q + b)
    q_len = tl.load(cu_q + b + 1) - q_begin
    if qt * BQ >= q_len:
        return
    k_len = tl.load(cu_k + b + 1) - tl.load(cu_k + b)
    page_cursor = tl.load(cu_pages + b)
    pages_b = (k_len + _PAGE_C - 1) // _PAGE_C

    offs_q = qt * BQ + tl.arange(0, BQ)
    q_valid = offs_q < q_len
    q_abs = q_begin + offs_q
    offs_b64 = tl.arange(0, _PB_C)
    offs_g = tl.arange(0, _G_C)
    offs_n = tl.arange(0, _PAGE_C)
    offs_t = tl.arange(0, TOPK)

    q_ptrs = q_fp4 + (q_abs[:, None] * heads_q + hq) * _PB_C + offs_b64[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None], other=0)
    qs_ptrs = q_scale + (q_abs[:, None] * heads_q + hq) * _G_C + offs_g[None, :]
    q_sc = tl.load(qs_ptrs, mask=q_valid[:, None], other=0)

    if has_qo_offset:
        offset = tl.load(qo_offset + b)
    else:
        offset = k_len - q_len

    kt_hi = pages_b
    if causal:
        q_tile_last = tl.minimum(qt * BQ + BQ - 1, q_len - 1)
        visible_limit = q_tile_last + offset
        if visible_limit < 0:
            return
        kt_hi = tl.minimum(pages_b, visible_limit // _PAGE_C + 1)

    # Running top-k: forced slots carry +inf so they are never evicted.
    top_s = tl.full([BQ, TOPK], float("-inf"), dtype=tl.float32)
    top_i = tl.full([BQ, TOPK], -1, dtype=tl.int32)
    if FORCE_DIAG:
        diag = (offs_q + offset) // _PAGE_C
        diag_ok = q_valid & (diag >= 0) & (diag < pages_b)
        top_s = tl.where((offs_t[None, :] == 0) & diag_ok[:, None], float("inf"), top_s)
        top_i = tl.where((offs_t[None, :] == 0) & diag_ok[:, None], diag[:, None].to(tl.int32), top_i)
    if FORCE_BEGIN > 0:
        for fb in tl.static_range(FORCE_BEGIN):
            slot = FORCE_DIAG + fb
            page_ok = q_valid & (fb < pages_b)
            # skip if this block already occupies a forced slot (diagonal)
            already = tl.sum(tl.where(top_i == fb, 1, 0), axis=1) > 0
            seed = page_ok & (~already)
            top_s = tl.where((offs_t[None, :] == slot) & seed[:, None], float("inf"), top_s)
            top_i = tl.where((offs_t[None, :] == slot) & seed[:, None], fb, top_i)

    for ktile in range(0, kt_hi):
        physical_page = tl.load(kv_indices + page_cursor + ktile)
        k_start = ktile * _PAGE_C
        tok_valid = k_start + offs_n < k_len
        k_ptrs = (
            k_fp4
            + ((physical_page * heads_k + hk) * _PAGE_C + offs_n[None, :]) * _PB_C
            + offs_b64[:, None]
        )
        k_tile = tl.load(k_ptrs, mask=tok_valid[None, :], other=0)
        ks_ptrs = (
            k_scale
            + ((physical_page * heads_k + hk) * _PAGE_C + offs_n[:, None]) * _G_C
            + offs_g[None, :]
        )
        k_sc = tl.load(ks_ptrs, mask=tok_valid[:, None], other=0)

        logits = tl.dot_scaled(
            q_tile, q_sc, "e2m1", k_tile, k_sc, "e2m1",
            lhs_k_pack=True, rhs_k_pack=True, out_dtype=tl.float32,
        )
        visible = q_valid[:, None] & tok_valid[None, :]
        if causal:
            visible = visible & (offs_q[:, None] + offset >= (k_start + offs_n)[None, :])
        logits = tl.where(visible, logits, -float("inf"))
        page_max = tl.max(logits, axis=1)
        row_has_visible = tl.max(visible.to(tl.int32), axis=1) > 0

        # Insertion: replace the first minimum slot when this page beats it.
        dup = tl.sum(tl.where(top_i == ktile, 1, 0), axis=1) > 0
        cand = tl.where(row_has_visible & (~dup), page_max, -float("inf"))
        row_min = tl.min(top_s, axis=1)
        is_min = top_s == row_min[:, None]
        first_min = is_min & (tl.cumsum(is_min.to(tl.int32), axis=1) == 1)
        take = (cand > row_min)[:, None] & first_min
        top_s = tl.where(take, cand[:, None], top_s)
        top_i = tl.where(take, ktile, top_i)

    # Epilogue: ascending block ids with -1 sorted to the tail.
    sort_key = tl.where(top_s == -float("inf"), 2147483647, top_i)
    sort_key = tl.sort(sort_key, dim=1)
    out = tl.where(sort_key == 2147483647, -1, sort_key)
    o_ptrs = q2k + (hq * total_q + q_abs[:, None]) * TOPK + offs_t[None, :]
    tl.store(o_ptrs, out.to(tl.int32), mask=q_valid[:, None])


def fp4_indexer_topk(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_page_offsets: torch.Tensor,
    *,
    topk: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    kv_indices: torch.Tensor,
    fp4_format: str = "mxfp4",
    causal: bool = True,
    qo_offset: Optional[torch.Tensor] = None,
    force_diagonal_blocks: int = 1,
    force_begin_blocks: int = 0,
) -> torch.Tensor:
    """Fused FP4 block scoring + top-k page selection.

    Returns ``q2k`` int32 of shape ``[Hq, total_q, topk]``: ascending block
    ids with -1 padding sorted to the tail — the MSA selection contract —
    without materializing the ``[Hq, tiles, total_q]`` score tensor or any
    of the downstream topk/mask/sort/gather launches.

    ``force_diagonal_blocks=1`` pins each query's own block (requires
    causal); ``force_begin_blocks=n`` pins blocks ``0..n-1`` (attention
    sinks). Forced slots consume top-k capacity.
    """
    if str(fp4_format).lower() != "mxfp4":
        raise NotImplementedError("fp4_indexer_topk supports fp4_format='mxfp4'")
    if topk not in (4, 8, 16, 32):
        raise ValueError(f"topk must be one of 4/8/16/32, got {topk}")
    if force_diagonal_blocks not in (0, 1):
        raise ValueError("force_diagonal_blocks must be 0 or 1")
    if force_diagonal_blocks and not causal:
        raise ValueError("force_diagonal_blocks requires causal=True")
    if force_diagonal_blocks + force_begin_blocks >= topk:
        raise ValueError("forced blocks must leave top-k capacity free")
    q_bytes = q_fp4.view(torch.uint8)
    k_bytes = k_fp4.view(torch.uint8)
    total_q, heads_q, packed = (int(v) for v in q_bytes.shape)
    page_count, heads_k, page_size, k_packed = (int(v) for v in k_bytes.shape)
    if packed != _PACKED_BYTES or k_packed != _PACKED_BYTES or page_size != _PAGE_SIZE:
        raise ValueError("FP4 indexer expects D=128 packed as 64 bytes and 128-token pages")
    if heads_q % heads_k != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    q_sc = q_scale.view(torch.uint8)
    k_sc = k_scale.view(torch.uint8)
    if tuple(q_sc.shape) != (total_q, heads_q, _MX_SCALE_GROUPS):
        raise ValueError("q_scale must have public layout [total_q, Hq, 4] for mxfp4")
    if tuple(k_sc.shape) != (page_count, heads_k, _PAGE_SIZE, _MX_SCALE_GROUPS):
        raise ValueError("k_scale must have public layout [pages, Hk, 128, 4] for mxfp4")
    batch = int(cu_seqlens_q.shape[0]) - 1
    q2k = torch.full((heads_q, total_q, topk), -1, dtype=torch.int32, device=q_fp4.device)
    if batch <= 0 or max_seqlen_k <= 0:
        return q2k
    max_q_tiles = (int(max_seqlen_q) + _BLOCK_Q - 1) // _BLOCK_Q
    if qo_offset is None:
        qo_offset_arg = cu_seqlens_q
        has_qo_offset = False
    else:
        qo_offset_arg = qo_offset
        has_qo_offset = True
    grid = (batch * max_q_tiles, heads_q)
    _fp4_indexer_topk_kernel[grid](
        q_bytes.contiguous(),
        k_bytes.contiguous(),
        q_sc.contiguous(),
        k_sc.contiguous(),
        cu_seqlens_q,
        cu_seqlens_k,
        cu_page_offsets,
        kv_indices,
        qo_offset_arg,
        q2k,
        int(total_q),
        int(heads_q),
        int(heads_k),
        int(heads_q // heads_k),
        int(max_q_tiles),
        bool(causal),
        bool(has_qo_offset),
        TOPK=int(topk),
        FORCE_DIAG=int(force_diagonal_blocks),
        FORCE_BEGIN=int(force_begin_blocks),
        BQ=_BLOCK_Q,
        num_warps=8,
        num_stages=2,
    )
    return q2k
