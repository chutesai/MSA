"""Q-stationary single-pass sparse attention for SM120 (MSA top-k block selection).

Design (replaces the two_phase / recompute CSR kernels for training):

* Forward: one program per (batch, tile of BLOCK_T consecutive query tokens,
  kv head).  The GQA query heads sharing the kv head are folded into the MMA
  M dimension (M = BLOCK_T * qhead_per_kv), so each selected K/V block is
  loaded once per tile instead of once per (16-query chunk, q_rep).  The
  program iterates over the union of KV blocks selected by its BLOCK_T tokens
  with a per-token selection mask and maintains an online softmax in
  registers.  No global partials, no atomics, no LSE pre-pass, no
  data-dependent grids (and therefore no host synchronization).

* Backward: standard FlashAttention-2 style split.  dQ is a Q-stationary
  kernel over the same block unions (single writer per output row).  dK/dV is
  a KV-stationary kernel over the existing K2Q CSR rows, sub-blocked to 64 kv
  tokens, accumulating in registers with no atomics anywhere: rows longer
  than FMHA_SM120_QSTAT_SPLIT_ROWS (an attention-sink block's row spans every
  query) are split across programs into per-split slabs and reduced in fixed
  order, so gradients stay deterministic.

* Precision: bf16 tensor-core math with fp32 accumulators by default.  The
  fp8 variant quantizes Q per-row to e4m3 inside the kernel, reads
  pre-quantized e4m3 K/V (per-token K scales, per-channel V scales) and
  quantizes P to e4m3 for the PV matmul, so every forward matmul is a native
  e4m3 x e4m3 tensor-core op on SM120.  All scales are applied outside the
  MMA.  Backward stays bf16 math with fp8 K/V reads (dequantized in
  registers), matching the FA3 precision recipe.

Constraints (checked by the wrapper, which callers should guard against):
  * head_dim == 128, blk_kv == 128
  * fixed-length unpacked batches (cu_seqlens strides equal); this matches
    MSAAttentionLayer, which already rejects true varlen input.
  * no paged KV (training path only; decode keeps the paged kernels).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
NEG_BIG = tl.constexpr(-1.0e30)


# ---------------------------------------------------------------------------
# Union-of-blocks builder (pure torch, no host sync)
# ---------------------------------------------------------------------------


def build_tile_block_union(
    q2k: torch.Tensor,
    batch: int,
    seq_len: int,
    num_blocks: int,
    block_t: int,
    with_selbits: bool = False,
):
    """Union of selected KV blocks per (kv_head, batch, query tile).

    Args:
        q2k: (H_kv, batch*seq_len, topk) int32, ascending valid block ids with
            -1 padding sorted to the end (the layout MSAAttentionLayer builds).
        with_selbits: also return per-token membership bitmasks for the
            hand-written CUDA forward (bit t of selbits[h, b, tile, u]: did
            tile-local token t select union entry u).
    Returns:
        union: (H_kv, batch, ntiles, U_max) int32 ascending block ids, padded
            with num_blocks sentinel past each row's count.
        counts: (H_kv, batch, ntiles) int32 number of valid union entries.
        selbits: (H_kv, batch, ntiles, U_max) int64, only when with_selbits.
    """
    head_kv, total_q, topk = q2k.shape
    if total_q != batch * seq_len:
        raise ValueError(f"q2k rows {total_q} != batch*seq_len {batch * seq_len}")
    ntiles = triton.cdiv(seq_len, block_t)
    pad_q = ntiles * block_t - seq_len
    sel = q2k.view(head_kv, batch, seq_len, topk)
    if pad_q:
        sel = torch.nn.functional.pad(sel, (0, 0, 0, pad_q), value=-1)
    sel = sel.reshape(head_kv, batch, ntiles, block_t * topk)
    # Bitmap over block ids; -1 padding routed to a trash column.
    bitmap = torch.zeros(
        (head_kv, batch, ntiles, num_blocks + 1),
        dtype=torch.int8,
        device=q2k.device,
    )
    idx = sel.long().clamp_min(-1)
    idx = torch.where(idx < 0, torch.full_like(idx, num_blocks), idx)
    bitmap.scatter_(-1, idx, 1)
    bitmap = bitmap[..., :num_blocks]
    counts = bitmap.sum(dim=-1, dtype=torch.int32)
    u_max = min(block_t * topk, num_blocks)
    # Stable descending argsort of {0,1} puts selected block ids first, in
    # ascending id order.  Sentinel (num_blocks) marks the padded tail.
    order = bitmap.argsort(dim=-1, descending=True, stable=True)[..., :u_max]
    position = torch.arange(u_max, device=q2k.device).view(1, 1, 1, u_max)
    union = torch.where(
        position < counts.unsqueeze(-1).long(),
        order,
        torch.full_like(order, num_blocks),
    ).to(torch.int32)
    if not with_selbits:
        return union.contiguous(), counts.contiguous()
    # Per-token membership over the same scatter indices: (token, block) pairs
    # are unique within a tile (q2k lists unique blocks per token), so summing
    # 1 << t equals bitwise OR.  The trash column swallows -1 padding and is
    # zeroed so sentinel union entries gather empty masks.
    weights = torch.ones(1, dtype=torch.int64, device=q2k.device) << (
        torch.arange(block_t * topk, device=q2k.device, dtype=torch.int64) // topk
    )
    bits = torch.zeros(
        (head_kv, batch, ntiles, num_blocks + 1),
        dtype=torch.int64,
        device=q2k.device,
    )
    bits.scatter_add_(-1, idx, weights.expand_as(idx))
    bits[..., num_blocks] = 0
    selbits = bits.gather(-1, union.long()).contiguous()
    return union.contiguous(), counts.contiguous(), selbits


def build_row_maps_fixed(
    batch: int, num_blocks: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """row -> (batch, kv block) maps for fixed-length batches, on-GPU.

    Matches the CSR row ordering used by build_k2q_csr/_build_row_maps,
    which is kv_block-major with batch as the minor axis.
    """
    rows = torch.arange(batch * num_blocks, device=device, dtype=torch.int32)
    return (rows % batch).contiguous(), (rows // batch).contiguous()


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------


@triton.jit
def _qstat_fwd_kernel(
    q,
    k,
    v,
    q2k,
    union_blocks,
    union_counts,
    k_scale,
    v_scale,
    out,
    lse_out,
    softmax_scale: tl.constexpr,
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    ntiles: tl.constexpr,
    num_blocks: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    u_max: tl.constexpr,
    KV_FP8: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLK_KV: tl.constexpr,
    DIM: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    kv_head = tl.program_id(1)
    b = pid_bt // ntiles
    tile = pid_bt % ntiles

    M: tl.constexpr = BLOCK_T * qhead_per_kv
    offs_m = tl.arange(0, M)
    t_of_m = offs_m % BLOCK_T
    g_of_m = offs_m // BLOCK_T
    offs_d = tl.arange(0, DIM)
    offs_n = tl.arange(0, BLK_KV)
    offs_k = tl.arange(0, topk)

    t0 = tile * BLOCK_T
    q_pos = t0 + t_of_m
    tok_valid = q_pos < seq_len
    gt_m = b * seq_len + q_pos
    qh_m = kv_head * qhead_per_kv + g_of_m

    q_ptrs = q + (gt_m[:, None] * head_q + qh_m[:, None]) * DIM + offs_d[None, :]

    if KV_FP8:
        # Per-row Q quantization to e4m3, fused with the (single) Q read; the
        # inverse scale is applied to scores after the native fp8 dot.  The
        # backward kernels re-read Q many times, so THEY get a pre-quantized
        # copy instead (built once in qstat_backward with identical math).
        q_f32 = tl.load(q_ptrs, mask=tok_valid[:, None], other=0.0).to(tl.float32)
        q_amax = tl.maximum(tl.max(tl.abs(q_f32), axis=1), 1e-8)
        q_op = (q_f32 * (448.0 / q_amax)[:, None]).to(tl.float8e4nv)
        q_deq = q_amax / 448.0
        v_ch_scale = tl.load(v_scale + kv_head * DIM + offs_d).to(tl.float32)
    else:
        q_tile = tl.load(q_ptrs, mask=tok_valid[:, None], other=0.0)
        q_op = q_tile.to(tl.bfloat16)

    q2k_ptrs = q2k + (kv_head * (batch * seq_len) + gt_m[:, None]) * topk + offs_k[None, :]
    q2k_m = tl.load(q2k_ptrs, mask=tok_valid[:, None], other=-1)

    union_base = union_blocks + ((kv_head * batch + b) * ntiles + tile) * u_max
    cnt = tl.load(union_counts + (kv_head * batch + b) * ntiles + tile)

    m_i = tl.full((M,), NEG_BIG, tl.float32)
    l_i = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, DIM), tl.float32)

    k_seq_base = b * seq_len
    for u in range(0, cnt):
        blk = tl.load(union_base + u)
        pos = blk * BLK_KV + offs_n
        kv_valid = pos < seq_len
        k_tok = k_seq_base + pos
        sel_m = tl.max((q2k_m == blk).to(tl.int32), axis=1) > 0

        k_ptrs = k + (k_tok[None, :] * head_kv + kv_head) * DIM + offs_d[:, None]
        if KV_FP8:
            k_tile = tl.load(k_ptrs, mask=kv_valid[None, :], other=0).to(
                tl.float8e4nv, bitcast=True
            )
            ks = tl.load(k_scale + k_tok * head_kv + kv_head, mask=kv_valid, other=0.0)
            s = tl.dot(q_op, k_tile, out_dtype=tl.float32)
            s = s * (q_deq[:, None] * ks[None, :]) * softmax_scale
        else:
            k_tile = tl.load(k_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.bfloat16)
            s = tl.dot(q_op, k_tile, out_dtype=tl.float32) * softmax_scale

        mask = sel_m[:, None] & tok_valid[:, None] & kv_valid[None, :]
        mask = mask & (pos[None, :] <= q_pos[:, None])
        s = tl.where(mask, s, NEG_BIG)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(mask, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v + (k_tok[:, None] * head_kv + kv_head) * DIM + offs_d[None, :]
        if KV_FP8:
            v_tile = tl.load(v_ptrs, mask=kv_valid[:, None], other=0).to(
                tl.float8e4nv, bitcast=True
            )
            p_op = (p * 448.0).to(tl.float8e4nv)
            acc += tl.dot(p_op, v_tile, out_dtype=tl.float32) * (1.0 / 448.0)
        else:
            v_tile = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.bfloat16)
            acc += tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)
        m_i = m_new

    if KV_FP8:
        acc = acc * v_ch_scale[None, :]

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o = acc / safe_l[:, None]
    o_ptrs = out + (gt_m[:, None] * head_q + qh_m[:, None]) * DIM + offs_d[None, :]
    tl.store(o_ptrs, o.to(out.dtype.element_ty), mask=tok_valid[:, None])
    # Empty rows (no selected blocks) emit a FINITE sentinel instead of
    # -inf: downstream lse consumers (branch merges, z-loss, logging)
    # stay finite, and exp(sentinel - x) underflows to exactly 0 so
    # merge weights are unchanged. Backward guards key on > -1e4.
    lse = tl.where(l_i > 0.0, m_i + tl.log(safe_l), -30000.0)
    tl.store(lse_out + gt_m * head_q + qh_m, lse, mask=tok_valid)


# ---------------------------------------------------------------------------
# Backward kernels
# ---------------------------------------------------------------------------


@triton.jit
def _qstat_delta_kernel(
    out,
    dout,
    delta,
    total_rows: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # delta[r] = sum_d out[r, d] * dout[r, d] in fp32, reading bf16 directly
    # (the torch equivalent materializes two full fp32 casts first).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_rows
    offs_d = tl.arange(0, DIM)
    o = tl.load(out + offs[:, None] * DIM + offs_d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(dout + offs[:, None] * DIM + offs_d[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    tl.store(delta + offs, tl.sum(o * do, axis=1), mask=mask)


def _compute_delta(out: torch.Tensor, dout: torch.Tensor) -> torch.Tensor:
    total_q, head_q, dim = out.shape
    rows = total_q * head_q
    delta = torch.empty((total_q, head_q), device=out.device, dtype=torch.float32)
    block = 64
    _qstat_delta_kernel[(triton.cdiv(rows, block),)](
        out, dout, delta, int(rows), DIM=int(dim), BLOCK=block, num_warps=4
    )
    return delta


@triton.jit
def _qstat_bwd_dq_kernel(
    q,
    k,
    v,
    q2k,
    union_blocks,
    union_counts,
    k_scale,
    v_scale,
    lse_in,
    delta,
    dout,
    dq,
    softmax_scale: tl.constexpr,
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    ntiles: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    u_max: tl.constexpr,
    KV_FP8: tl.constexpr,
    GRAD_FP8: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLK_KV: tl.constexpr,
    DIM: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    kv_head = tl.program_id(1)
    b = pid_bt // ntiles
    tile = pid_bt % ntiles

    M: tl.constexpr = BLOCK_T * qhead_per_kv
    offs_m = tl.arange(0, M)
    t_of_m = offs_m % BLOCK_T
    g_of_m = offs_m // BLOCK_T
    offs_d = tl.arange(0, DIM)
    offs_n = tl.arange(0, BLK_KV)
    offs_k = tl.arange(0, topk)

    t0 = tile * BLOCK_T
    q_pos = t0 + t_of_m
    tok_valid = q_pos < seq_len
    gt_m = b * seq_len + q_pos
    qh_m = kv_head * qhead_per_kv + g_of_m

    q_ptrs = q + (gt_m[:, None] * head_q + qh_m[:, None]) * DIM + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=tok_valid[:, None], other=0.0)
    do_ptrs = dout + (gt_m[:, None] * head_q + qh_m[:, None]) * DIM + offs_d[None, :]
    do_raw = tl.load(do_ptrs, mask=tok_valid[:, None], other=0.0)
    do_tile = do_raw.to(tl.bfloat16)
    lse_m = tl.load(lse_in + gt_m * head_q + qh_m, mask=tok_valid, other=-float("inf"))
    dl_m = tl.load(delta + gt_m * head_q + qh_m, mask=tok_valid, other=0.0)
    lse_finite = lse_m > -1.0e4  # sentinel-aware (empty rows = -30000)
    lse_safe = tl.where(lse_finite, lse_m, 0.0)

    if KV_FP8:
        q_f32 = q_tile.to(tl.float32)
        q_amax = tl.maximum(tl.max(tl.abs(q_f32), axis=1), 1e-8)
        q_op = (q_f32 * (448.0 / q_amax)[:, None]).to(tl.float8e4nv)
        q_deq = q_amax / 448.0
        v_ch_scale = tl.load(v_scale + kv_head * DIM + offs_d).to(tl.float32)
        if GRAD_FP8:
            # dP = dO . V^T sums over channels d, so V's per-channel scale
            # sits inside the sum: fold it into dO before quantizing, then
            # apply dO's per-row scale after the (native fp8) dot.
            do_v = do_raw.to(tl.float32) * v_ch_scale[None, :]
            dov_amax = tl.maximum(tl.max(tl.abs(do_v), axis=1), 1e-8)
            do_v_q = (do_v * (448.0 / dov_amax)[:, None]).to(tl.float8e4nv)
            dov_deq = dov_amax / 448.0
    else:
        q_op = q_tile.to(tl.bfloat16)

    q2k_ptrs = q2k + (kv_head * (batch * seq_len) + gt_m[:, None]) * topk + offs_k[None, :]
    q2k_m = tl.load(q2k_ptrs, mask=tok_valid[:, None], other=-1)

    union_base = union_blocks + ((kv_head * batch + b) * ntiles + tile) * u_max
    cnt = tl.load(union_counts + (kv_head * batch + b) * ntiles + tile)

    dq_acc = tl.zeros((M, DIM), tl.float32)
    k_seq_base = b * seq_len
    for u in range(0, cnt):
        blk = tl.load(union_base + u)
        pos = blk * BLK_KV + offs_n
        kv_valid = pos < seq_len
        k_tok = k_seq_base + pos
        sel_m = tl.max((q2k_m == blk).to(tl.int32), axis=1) > 0

        k_ptrs = k + (k_tok[None, :] * head_kv + kv_head) * DIM + offs_d[:, None]
        v_ptrs = v + (k_tok[None, :] * head_kv + kv_head) * DIM + offs_d[:, None]
        if KV_FP8:
            k_f8 = tl.load(k_ptrs, mask=kv_valid[None, :], other=0).to(
                tl.float8e4nv, bitcast=True
            )
            ks = tl.load(k_scale + k_tok * head_kv + kv_head, mask=kv_valid, other=0.0)
            s = tl.dot(q_op, k_f8, out_dtype=tl.float32)
            s = s * (q_deq[:, None] * ks[None, :]) * softmax_scale
            k_bf = (k_f8.to(tl.float32) * ks[None, :]).to(tl.bfloat16)
            v_f8 = tl.load(v_ptrs, mask=kv_valid[None, :], other=0).to(
                tl.float8e4nv, bitcast=True
            )
        else:
            k_bf = tl.load(k_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.bfloat16)
            v_bf = tl.load(v_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.bfloat16)
            s = tl.dot(q_op, k_bf, out_dtype=tl.float32) * softmax_scale

        mask = sel_m[:, None] & tok_valid[:, None] & kv_valid[None, :]
        mask = mask & (pos[None, :] <= q_pos[:, None])
        p = tl.exp(tl.where(mask, s, NEG_BIG) - lse_safe[:, None])
        p = tl.where(mask & lse_finite[:, None], p, 0.0)

        if KV_FP8 and GRAD_FP8:
            dp = tl.dot(do_v_q, v_f8, out_dtype=tl.float32) * dov_deq[:, None]
        elif KV_FP8:
            v_bf = (v_f8.to(tl.float32) * v_ch_scale[:, None]).to(tl.bfloat16)
            dp = tl.dot(do_tile, v_bf, out_dtype=tl.float32)
        else:
            dp = tl.dot(do_tile, v_bf, out_dtype=tl.float32)
        ds = p * (dp - dl_m[:, None])
        dq_acc += tl.dot(ds.to(tl.bfloat16), tl.trans(k_bf), out_dtype=tl.float32)

    dq_acc = dq_acc * softmax_scale
    dq_ptrs = dq + (gt_m[:, None] * head_q + qh_m[:, None]) * DIM + offs_d[None, :]
    tl.store(dq_ptrs, dq_acc.to(dq.dtype.element_ty), mask=tok_valid[:, None])


@triton.jit
def _qstat_bwd_dkdv_kernel(
    q,
    k,
    v,
    k2q_row_ptr,
    k2q_q_indices,
    row_batch,
    row_kv_block,
    k_scale,
    v_scale,
    lse_in,
    delta,
    dout,
    dk,
    dv,
    grad_split_stride,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    seq_len: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    KV_FP8: tl.constexpr,
    GRAD_FP8: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_TQ: tl.constexpr,
    BLK_KV: tl.constexpr,
    SUB_N: tl.constexpr,
    DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_idx = tl.program_id(2)
    nsub: tl.constexpr = BLK_KV // SUB_N
    row = pid // nsub
    sub = pid % nsub

    M: tl.constexpr = BLOCK_TQ * qhead_per_kv
    offs_m = tl.arange(0, M)
    t_of_m = offs_m % BLOCK_TQ
    g_of_m = offs_m // BLOCK_TQ
    offs_d = tl.arange(0, DIM)
    offs_n = tl.arange(0, SUB_N)

    b = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    pos = kv_block * BLK_KV + sub * SUB_N + offs_n
    kv_valid = pos < seq_len
    k_tok = b * seq_len + pos

    k_ptrs = k + (k_tok[None, :] * head_kv + kv_head) * DIM + offs_d[:, None]
    v_ptrs = v + (k_tok[None, :] * head_kv + kv_head) * DIM + offs_d[:, None]
    if KV_FP8:
        ks = tl.load(k_scale + k_tok * head_kv + kv_head, mask=kv_valid, other=0.0)
        k_f8 = tl.load(k_ptrs, mask=kv_valid[None, :], other=0).to(tl.float8e4nv, bitcast=True)
        k_dn = (k_f8.to(tl.float32) * ks[None, :]).to(tl.bfloat16)
        v_ch_scale = tl.load(v_scale + kv_head * DIM + offs_d).to(tl.float32)
        v_f8 = tl.load(v_ptrs, mask=kv_valid[None, :], other=0).to(tl.float8e4nv, bitcast=True)
        v_dn = (v_f8.to(tl.float32) * v_ch_scale[:, None]).to(tl.bfloat16)
    else:
        k_dn = tl.load(k_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.bfloat16)
        v_dn = tl.load(v_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.bfloat16)

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    row_count = row_end - row_start

    # Long rows (e.g. an attention-sink block selected by every query) are
    # split across program_id(2) into disjoint BLOCK_TQ-aligned chunk ranges;
    # each split writes its own slab and the host sums the slabs in a fixed
    # order, so the result stays deterministic.
    if NSPLIT > 1:
        span = ((row_count + BLOCK_TQ - 1) // BLOCK_TQ + NSPLIT - 1) // NSPLIT * BLOCK_TQ
        chunk_lo = split_idx * span
        chunk_hi = tl.minimum(row_count, chunk_lo + span)
    else:
        chunk_lo = 0
        chunk_hi = row_count

    dk_acc = tl.zeros((SUB_N, DIM), tl.float32)
    dv_acc = tl.zeros((SUB_N, DIM), tl.float32)

    for chunk in range(chunk_lo, chunk_hi, BLOCK_TQ):
        csr_offs = row_start + chunk + t_of_m
        q_local = tl.load(
            k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
            mask=csr_offs < row_end,
            other=-1,
        )
        q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < seq_len)
        q_global = b * seq_len + tl.where(q_valid, q_local, 0)
        qh_m = kv_head * qhead_per_kv + g_of_m

        q_ptrs = q + (q_global[:, None] * head_q + qh_m[:, None]) * DIM + offs_d[None, :]
        if KV_FP8:
            # Requantize Q exactly as the forward kernel did so that
            # p = exp(s - lse) uses the same scores the saved LSE normalizes.
            q_f32 = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0).to(tl.float32)
            q_amax = tl.maximum(tl.max(tl.abs(q_f32), axis=1), 1e-8)
            q_f8 = (q_f32 * (448.0 / q_amax)[:, None]).to(tl.float8e4nv)
            q_deq = q_amax / 448.0
            q_m = (q_f8.to(tl.float32) * q_deq[:, None]).to(tl.bfloat16)
        else:
            q_m = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0).to(tl.bfloat16)
        do_ptrs = dout + (q_global[:, None] * head_q + qh_m[:, None]) * DIM + offs_d[None, :]
        do_raw = tl.load(do_ptrs, mask=q_valid[:, None], other=0.0)
        do_m = do_raw.to(tl.bfloat16)
        lse_m = tl.load(
            lse_in + q_global * head_q + qh_m, mask=q_valid, other=-float("inf")
        )
        dl_m = tl.load(delta + q_global * head_q + qh_m, mask=q_valid, other=0.0)
        lse_finite = lse_m > -1.0e4  # sentinel-aware (empty rows = -30000)
        lse_safe = tl.where(lse_finite, lse_m, 0.0)

        if KV_FP8:
            s = tl.dot(q_f8, k_f8, out_dtype=tl.float32)
            s = s * (q_deq[:, None] * ks[None, :]) * softmax_scale
        else:
            s = tl.dot(q_m, k_dn, out_dtype=tl.float32) * softmax_scale
        mask = q_valid[:, None] & kv_valid[None, :] & (pos[None, :] <= q_local[:, None])
        p = tl.exp(tl.where(mask, s, NEG_BIG) - lse_safe[:, None])
        p = tl.where(mask & lse_finite[:, None], p, 0.0)

        if KV_FP8 and GRAD_FP8:
            # dP = dO . V^T: fold V's per-channel scale into dO pre-quant
            # (it sits inside the d-sum), apply dO's row scale post-dot.
            do_f32 = do_raw.to(tl.float32)
            do_v = do_f32 * v_ch_scale[None, :]
            dov_amax = tl.maximum(tl.max(tl.abs(do_v), axis=1), 1e-8)
            do_v_q = (do_v * (448.0 / dov_amax)[:, None]).to(tl.float8e4nv)
            dp = tl.dot(do_v_q, v_f8, out_dtype=tl.float32) * (dov_amax / 448.0)[:, None]
        else:
            dp = tl.dot(do_m, v_dn, out_dtype=tl.float32)
        ds = p * (dp - dl_m[:, None])
        ds_bf = ds.to(tl.bfloat16)
        dk_acc += tl.dot(tl.trans(ds_bf), q_m, out_dtype=tl.float32)

        if KV_FP8 and GRAD_FP8:
            # dV = P^T . dO: dO's per-row scale sits inside the m-sum, so
            # fold it into P before quantizing; a single per-chunk scale
            # lifts the folded P into e4m3 range and is applied post-dot.
            do_amax = tl.maximum(tl.max(tl.abs(do_f32), axis=1), 1e-8)
            do_q = (do_f32 * (448.0 / do_amax)[:, None]).to(tl.float8e4nv)
            do_deq = do_amax / 448.0
            c_max = tl.maximum(tl.max(do_deq, axis=0), 1e-20)
            p_do = p * do_deq[:, None]
            p_do_q = (p_do * (448.0 / c_max)).to(tl.float8e4nv)
            dv_acc += tl.dot(tl.trans(p_do_q), do_q, out_dtype=tl.float32) * (
                c_max / 448.0
            )
        else:
            p_bf = p.to(tl.bfloat16)
            dv_acc += tl.dot(tl.trans(p_bf), do_m, out_dtype=tl.float32)

    dk_acc = dk_acc * softmax_scale
    split_base = split_idx * grad_split_stride
    dk_ptrs = dk + split_base + (k_tok[:, None] * head_kv + kv_head) * DIM + offs_d[None, :]
    dv_ptrs = dv + split_base + (k_tok[:, None] * head_kv + kv_head) * DIM + offs_d[None, :]
    tl.store(dk_ptrs, dk_acc.to(dk.dtype.element_ty), mask=kv_valid[:, None])
    tl.store(dv_ptrs, dv_acc.to(dv.dtype.element_ty), mask=kv_valid[:, None])


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _pick_block_t(qhead_per_kv: int) -> int:
    # Keep the M tile at exactly 64 rows (register budget / MMA shape sweet
    # spot).  Non-power-of-two GQA groups are unsupported: fail loudly rather
    # than emit an odd M tile.
    if qhead_per_kv not in SUPPORTED_QHEAD_PER_KV:
        raise ValueError(
            f"qstat backend supports qhead_per_kv in {SUPPORTED_QHEAD_PER_KV}, "
            f"got {qhead_per_kv}"
        )
    return {1: 64, 2: 32, 4: 16, 8: 8, 16: 4}[qhead_per_kv]


def _fixed_geometry(cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, total_q: int):
    """Batch/seq_len for fixed-length batches without device sync.

    NOTE: this intentionally does NOT read cu_seqlens contents (that would
    force a host sync).  Callers are responsible for guaranteeing equal-length
    unpacked batches — MSAAttentionLayer validates this before dispatch.
    Passing true varlen metadata here produces silently wrong indexing.
    """
    batch = int(cu_seqlens_q.numel() - 1)
    if batch <= 0 or total_q % batch != 0:
        raise ValueError("qstat path requires fixed-length unpacked batches")
    if cu_seqlens_k.numel() != cu_seqlens_q.numel():
        raise ValueError("qstat path requires cu_seqlens_q == cu_seqlens_k")
    return batch, total_q // batch


# GQA group sizes the tiling has been validated for (M = BLOCK_T * group must
# be a power-of-two multiple of 16, capped at 64 rows).
SUPPORTED_QHEAD_PER_KV = (1, 2, 4, 8, 16)


def qstat_forward(
    q: torch.Tensor,
    k_op: torch.Tensor,
    v_op: torch.Tensor,
    q2k: torch.Tensor,
    union: torch.Tensor,
    counts: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    num_blocks: int,
    block_t: int,
    topk: int,
    blk_kv: int,
    softmax_scale: float,
    kv_fp8: bool,
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, head_q, dim = q.shape
    head_kv = int(q2k.shape[0])
    qhead_per_kv = head_q // head_kv
    ntiles = triton.cdiv(seq_len, block_t)
    u_max = int(union.shape[-1])
    out = torch.empty((total_q, head_q, dim), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    dummy = counts  # placeholder pointer for unused scale args
    grid = (batch * ntiles, head_kv)
    m_rows = block_t * qhead_per_kv
    _qstat_fwd_kernel[grid](
        q,
        k_op,
        v_op,
        q2k,
        union,
        counts,
        k_scale if kv_fp8 else dummy,
        v_scale if kv_fp8 else dummy,
        out,
        lse,
        float(softmax_scale),
        int(batch),
        int(seq_len),
        int(ntiles),
        int(num_blocks),
        int(head_q),
        int(head_kv),
        int(qhead_per_kv),
        int(topk),
        int(u_max),
        bool(kv_fp8),
        BLOCK_T=int(block_t),
        BLK_KV=int(blk_kv),
        DIM=int(dim),
        num_warps=8 if m_rows >= 64 else 4,
        num_stages=1,
    )
    return out, lse


def qstat_backward(
    q: torch.Tensor,
    k_op: torch.Tensor,
    v_op: torch.Tensor,
    q2k: torch.Tensor,
    union: torch.Tensor,
    counts: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    num_blocks: int,
    block_t: int,
    topk: int,
    blk_kv: int,
    softmax_scale: float,
    kv_fp8: bool,
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    kv_grad_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_q, head_q, dim = q.shape
    head_kv = int(q2k.shape[0])
    qhead_per_kv = head_q // head_kv
    ntiles = triton.cdiv(seq_len, block_t)
    u_max = int(union.shape[-1])
    total_rows = batch * num_blocks

    dout = dout.contiguous()
    delta = _compute_delta(out, dout)  # (total_q, head_q) fp32

    # Opt-in native-fp8 gradient matmuls (dP and dV; dS-based dQ/dK stay
    # bf16).  NOTE: measured SLOWER than bf16 grads on SM120 (the backward is
    # gather/memory-bound, so extra quantization work outweighs the 2x MMA
    # rate); kept as an experimental probe only.
    grad_fp8 = kv_fp8 and os.environ.get("FMHA_SM120_QSTAT_FP8_BWD", "0") == "1"

    # dK/dV work per program is proportional to its CSR row length, and an
    # attention-sink block's row can hold every query in the batch, leaving a
    # handful of straggler programs to serialize the whole kernel. Split rows
    # longer than FMHA_SM120_QSTAT_SPLIT_ROWS across program_id(2), with a
    # fixed-order slab reduction to keep gradients deterministic. The one
    # row-length readback below is the only host sync on the qstat path.
    row_counts = k2q_row_ptr[:, 1:] - k2q_row_ptr[:, :-1]
    max_row_count = int(row_counts.max().item()) if row_counts.numel() else 0
    split_rows = int(os.environ.get("FMHA_SM120_QSTAT_SPLIT_ROWS", "2048"))
    nsplit = 1
    if split_rows > 0 and max_row_count > split_rows:
        nsplit = min(8, -(-max_row_count // split_rows))

    dq = torch.empty_like(q)
    if nsplit > 1:
        dk_out = torch.empty((nsplit, total_q, head_kv, dim), device=q.device, dtype=torch.float32)
        dv_out = torch.empty_like(dk_out)
        grad_split_stride = total_q * head_kv * dim
    else:
        dk_out = torch.empty((total_q, head_kv, dim), device=q.device, dtype=kv_grad_dtype)
        dv_out = torch.empty((total_q, head_kv, dim), device=q.device, dtype=kv_grad_dtype)
        grad_split_stride = 0
    dummy = counts

    m_rows = block_t * qhead_per_kv
    grid_dq = (batch * ntiles, head_kv)
    _qstat_bwd_dq_kernel[grid_dq](
        q,
        k_op,
        v_op,
        q2k,
        union,
        counts,
        k_scale if kv_fp8 else dummy,
        v_scale if kv_fp8 else dummy,
        lse,
        delta,
        dout,
        dq,
        float(softmax_scale),
        int(batch),
        int(seq_len),
        int(ntiles),
        int(head_q),
        int(head_kv),
        int(qhead_per_kv),
        int(topk),
        int(u_max),
        bool(kv_fp8),
        bool(grad_fp8),
        BLOCK_T=int(block_t),
        BLK_KV=int(blk_kv),
        DIM=int(dim),
        num_warps=8 if m_rows >= 64 else 4,
        num_stages=1,
    )

    row_batch, row_kv_block = build_row_maps_fixed(batch, num_blocks, q.device)
    sub_n = 64 if blk_kv >= 128 else blk_kv
    nsub = blk_kv // sub_n
    block_tq = _pick_block_t(qhead_per_kv)
    grid_dkdv = (total_rows * nsub, head_kv, nsplit)
    _qstat_bwd_dkdv_kernel[grid_dkdv](
        q,
        k_op,
        v_op,
        k2q_row_ptr,
        k2q_q_indices,
        row_batch,
        row_kv_block,
        k_scale if kv_fp8 else dummy,
        v_scale if kv_fp8 else dummy,
        lse,
        delta,
        dout,
        dk_out,
        dv_out,
        int(grad_split_stride),
        float(softmax_scale),
        int(total_q),
        int(total_rows),
        int(seq_len),
        int(head_q),
        int(head_kv),
        int(qhead_per_kv),
        int(topk),
        bool(kv_fp8),
        bool(grad_fp8),
        NSPLIT=int(nsplit),
        BLOCK_TQ=int(block_tq),
        BLK_KV=int(blk_kv),
        SUB_N=int(sub_n),
        DIM=int(dim),
        num_warps=8,
        # Two stages software-pipeline the per-chunk Q/dO gathers against the
        # MMAs (the K/V tiles are loop-invariant); measured ~5% on the step.
        num_stages=2,
    )
    if nsplit > 1:
        dk = dk_out.sum(dim=0).to(kv_grad_dtype)
        dv = dv_out.sum(dim=0).to(kv_grad_dtype)
    else:
        dk, dv = dk_out, dv_out
    return dq, dk, dv


class _QstatSparseAttention(torch.autograd.Function):
    """BF16 Q-stationary sparse attention (dense bf16 K/V inputs)."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q2k: torch.Tensor,
        k2q_row_ptr: torch.Tensor,
        k2q_q_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float,
        topk: int,
        blk_kv: int,
    ):
        total_q, head_q, dim = q.shape
        batch, seq_len = _fixed_geometry(cu_seqlens_q, cu_seqlens_k, total_q)
        num_blocks = triton.cdiv(seq_len, blk_kv)
        head_kv = int(q2k.shape[0])
        block_t = _pick_block_t(head_q // head_kv)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        impl = os.environ.get("FMHA_SM120_QSTAT_IMPL", "triton").strip().lower()
        if impl not in {"triton", "cuda"}:
            raise ValueError(f"FMHA_SM120_QSTAT_IMPL must be triton or cuda, got {impl!r}")
        use_cuda = impl == "cuda" and blk_kv == 128 and dim == 128
        if use_cuda:
            union, counts, selbits = build_tile_block_union(
                q2k, batch, seq_len, num_blocks, block_t, with_selbits=True
            )
            from src.sm120.qstat_cuda import qstat_forward_cuda

            out, lse = qstat_forward_cuda(
                q, k, v, union, counts, selbits,
                batch=batch, seq_len=seq_len, block_t=block_t,
                softmax_scale=softmax_scale,
            )
        else:
            union, counts = build_tile_block_union(
                q2k, batch, seq_len, num_blocks, block_t
            )
            out, lse = qstat_forward(
                q,
                k,
                v,
                q2k,
                union,
                counts,
                batch=batch,
                seq_len=seq_len,
                num_blocks=num_blocks,
                block_t=block_t,
                topk=topk,
                blk_kv=blk_kv,
                softmax_scale=softmax_scale,
                kv_fp8=False,
                k_scale=None,
                v_scale=None,
            )
        ctx.save_for_backward(
            q, k, v, q2k, union, counts, k2q_row_ptr, k2q_q_indices, out, lse
        )
        ctx.geom = (batch, seq_len, num_blocks, block_t, topk, blk_kv)
        ctx.softmax_scale = float(softmax_scale)
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, _dlse: Optional[torch.Tensor]):
        (
            q,
            k,
            v,
            q2k,
            union,
            counts,
            k2q_row_ptr,
            k2q_q_indices,
            out,
            lse,
        ) = ctx.saved_tensors
        batch, seq_len, num_blocks, block_t, topk, blk_kv = ctx.geom
        dq, dk, dv = qstat_backward(
            q,
            k,
            v,
            q2k,
            union,
            counts,
            k2q_row_ptr,
            k2q_q_indices,
            out,
            lse,
            dout,
            batch=batch,
            seq_len=seq_len,
            num_blocks=num_blocks,
            block_t=block_t,
            topk=topk,
            blk_kv=blk_kv,
            softmax_scale=ctx.softmax_scale,
            kv_fp8=False,
            k_scale=None,
            v_scale=None,
            kv_grad_dtype=k.dtype,
        )
        return (dq, dk, dv) + (None,) * 8


class _QstatSparseAttentionFp8(torch.autograd.Function):
    """Native-FP8 Q-stationary sparse attention.

    K/V arrive pre-quantized as uint8 views of e4m3 with per-token K scales
    (total_k, head_kv) fp32 and per-channel V scales (head_kv, dim) fp32.
    ``k_ref``/``v_ref`` are zero-stride logical stubs used purely as gradient
    edges (straight-through to the pre-quantization K/V).
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k_ref: torch.Tensor,
        v_ref: torch.Tensor,
        k_fp8_u8: torch.Tensor,
        v_fp8_u8: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        q2k: torch.Tensor,
        k2q_row_ptr: torch.Tensor,
        k2q_q_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float,
        topk: int,
        blk_kv: int,
    ):
        total_q, head_q, dim = q.shape
        batch, seq_len = _fixed_geometry(cu_seqlens_q, cu_seqlens_k, total_q)
        num_blocks = triton.cdiv(seq_len, blk_kv)
        head_kv = int(q2k.shape[0])
        block_t = _pick_block_t(head_q // head_kv)
        q = q.contiguous()
        k_fp8_u8 = k_fp8_u8.contiguous()
        v_fp8_u8 = v_fp8_u8.contiguous()
        k_scale = k_scale.contiguous().float()
        v_scale = v_scale.contiguous().float()
        impl = os.environ.get("FMHA_SM120_QSTAT_IMPL", "triton").strip().lower()
        if impl not in {"triton", "cuda"}:
            raise ValueError(f"FMHA_SM120_QSTAT_IMPL must be triton or cuda, got {impl!r}")
        use_cuda = (
            impl == "cuda" and blk_kv == 128 and dim == 128 and seq_len % 16 == 0
        )
        if use_cuda:
            union, counts, selbits = build_tile_block_union(
                q2k, batch, seq_len, num_blocks, block_t, with_selbits=True
            )
            from src.sm120.qstat_cuda import qstat_forward_fp8_cuda

            out, lse = qstat_forward_fp8_cuda(
                q, k_fp8_u8, v_fp8_u8, k_scale, v_scale, union, counts, selbits,
                batch=batch, seq_len=seq_len, block_t=block_t,
                softmax_scale=softmax_scale,
            )
        else:
            union, counts = build_tile_block_union(
                q2k, batch, seq_len, num_blocks, block_t
            )
            out, lse = qstat_forward(
                q,
                k_fp8_u8,
                v_fp8_u8,
                q2k,
                union,
                counts,
                batch=batch,
                seq_len=seq_len,
                num_blocks=num_blocks,
                block_t=block_t,
                topk=topk,
                blk_kv=blk_kv,
                softmax_scale=softmax_scale,
                kv_fp8=True,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        ctx.save_for_backward(
            q,
            k_fp8_u8,
            v_fp8_u8,
            k_scale,
            v_scale,
            q2k,
            union,
            counts,
            k2q_row_ptr,
            k2q_q_indices,
            out,
            lse,
        )
        ctx.geom = (batch, seq_len, num_blocks, block_t, topk, blk_kv)
        ctx.softmax_scale = float(softmax_scale)
        ctx.kv_grad_dtype = k_ref.dtype
        ctx.q_dtype = q.dtype
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, _dlse: Optional[torch.Tensor]):
        (
            q,
            k_fp8_u8,
            v_fp8_u8,
            k_scale,
            v_scale,
            q2k,
            union,
            counts,
            k2q_row_ptr,
            k2q_q_indices,
            out,
            lse,
        ) = ctx.saved_tensors
        batch, seq_len, num_blocks, block_t, topk, blk_kv = ctx.geom
        grads_impl = os.environ.get("FMHA_SM120_QSTAT_GRADS", "bf16").strip().lower()
        if grads_impl not in {"bf16", "fp8"}:
            raise ValueError(
                f"FMHA_SM120_QSTAT_GRADS must be bf16 or fp8, got {grads_impl!r}"
            )
        if grads_impl == "fp8" and blk_kv == 128 and q.shape[-1] == 128 and seq_len % 16 == 0:
            # EXPERIMENTAL full-e4m3 backward — NOT validated for production
            # training: a real run NaN'd at ~step 1000 (d1024, lr 3e-4
            # warmup, 8xDDP) with these gradients; per-row e4m3 amax scaling
            # is fragile under heavy-tailed gradient outliers. Do not adopt
            # without a >=2k-step loss A/B at your exact config. An
            # e5m2-gradient rework is the planned path to re-qualification.
            import warnings

            warnings.warn(
                "FMHA_SM120_QSTAT_GRADS=fp8 is EXPERIMENTAL and has produced "
                "NaNs in real training (step ~1k). Use bf16 grads for "
                "production; gate any fp8-grads adoption on a >=2k-step "
                "loss A/B.",
                stacklevel=2,
            )
            from src.sm120.qstat_cuda import qstat_backward_fp8_cuda

            _, _, selbits = build_tile_block_union(
                q2k, batch, seq_len, num_blocks, block_t, with_selbits=True
            )
            row_batch, row_kv_block = build_row_maps_fixed(
                batch, num_blocks, q.device
            )
            dq, dk, dv = qstat_backward_fp8_cuda(
                q, k_fp8_u8, v_fp8_u8, k_scale, v_scale,
                dout.contiguous(), out, lse, union, counts, selbits,
                k2q_row_ptr, k2q_q_indices, row_batch, row_kv_block,
                batch=batch, seq_len=seq_len, block_t=block_t,
                block_tq=block_t, topk=topk,
                softmax_scale=ctx.softmax_scale,
                kv_grad_dtype=ctx.kv_grad_dtype,
            )
            return (dq, dk, dv) + (None,) * 12
        # Dequantize K/V once and run the bf16 backward kernels: the fp8
        # variants dequantize in registers before every MMA, which costs
        # ~10% of the train step. Numerics are unchanged — the same bf16
        # values reach the same kernels either way.
        k_deq = (
            k_fp8_u8.view(torch.float8_e4m3fn).float() * k_scale.unsqueeze(-1)
        ).to(ctx.q_dtype)
        v_deq = (
            v_fp8_u8.view(torch.float8_e4m3fn).float() * v_scale.unsqueeze(0)
        ).to(ctx.q_dtype)
        dq, dk, dv = qstat_backward(
            q,
            k_deq,
            v_deq,
            q2k,
            union,
            counts,
            k2q_row_ptr,
            k2q_q_indices,
            out,
            lse,
            dout,
            batch=batch,
            seq_len=seq_len,
            num_blocks=num_blocks,
            block_t=block_t,
            topk=topk,
            blk_kv=blk_kv,
            softmax_scale=ctx.softmax_scale,
            kv_fp8=False,
            k_scale=None,
            v_scale=None,
            kv_grad_dtype=ctx.kv_grad_dtype,
        )
        return (dq, dk, dv) + (None,) * 12


def sparse_attention_qstat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    *,
    topk: int,
    blk_kv: int = 128,
    softmax_scale: Optional[float] = None,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    return_softmax_lse: bool = False,
):
    """BF16 Q-stationary MSA sparse attention (training fwd+bwd)."""
    scale = float(softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5)
    out, lse = _QstatSparseAttention.apply(
        q,
        k,
        v,
        q2k_indices.contiguous(),
        k2q_row_ptr,
        k2q_q_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
        int(topk),
        int(blk_kv),
    )
    if return_softmax_lse:
        return out, lse
    return out


def sparse_attention_qstat_fp8(
    q: torch.Tensor,
    k_ref: torch.Tensor,
    v_ref: torch.Tensor,
    k_fp8_u8: torch.Tensor,
    v_fp8_u8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_indices: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    *,
    topk: int,
    blk_kv: int = 128,
    softmax_scale: Optional[float] = None,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    return_softmax_lse: bool = False,
):
    """Native-FP8 Q-stationary MSA sparse attention (training fwd+bwd)."""
    scale = float(softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5)
    out, lse = _QstatSparseAttentionFp8.apply(
        q,
        k_ref,
        v_ref,
        k_fp8_u8,
        v_fp8_u8,
        k_scale,
        v_scale,
        q2k_indices.contiguous(),
        k2q_row_ptr,
        k2q_q_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
        int(topk),
        int(blk_kv),
    )
    if return_softmax_lse:
        return out, lse
    return out


def quantize_kv_fp8_scaled(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize K/V (total, H_kv, D) to e4m3 with honest scales.

    K: per-token-per-head amax scale (applied as a score column-scale in the
    kernel).  V: per-channel-per-head amax scale (applied to the output).
    Returns (k_u8, v_u8, k_scale fp32 (total, H_kv), v_scale fp32 (H_kv, D)).
    """
    k_f = k.float()
    v_f = v.float()
    k_amax = k_f.abs().amax(dim=-1).clamp_min(1e-6)  # (total, H)
    v_amax = v_f.abs().amax(dim=0).clamp_min(1e-6)  # (H, D)
    k_q = (k_f * (FP8_MAX / k_amax).unsqueeze(-1)).to(torch.float8_e4m3fn)
    v_q = (v_f * (FP8_MAX / v_amax).unsqueeze(0)).to(torch.float8_e4m3fn)
    return (
        k_q.view(torch.uint8),
        v_q.view(torch.uint8),
        (k_amax / FP8_MAX).contiguous(),
        (v_amax / FP8_MAX).contiguous(),
    )


# ---------------------------------------------------------------------------
# MXFP4 (e2m1 + ue8m0/32) K/V cache format + v2 paged decode
# ---------------------------------------------------------------------------

_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize (..., D) to MXFP4: packed e2m1 nibbles + ue8m0 per-32 scales.

    Returns (packed (..., D//2) uint8, scales (..., D//32) uint8).  The
    layout matches SM120's native mxf4 operand format (element pairs packed
    low-nibble-first along the last axis, one shared power-of-two scale per
    32 consecutive elements).
    """
    if x.shape[-1] % 32 != 0:
        raise ValueError("MXFP4 requires the last dim to be a multiple of 32")
    xf = x.float()
    groups = xf.unflatten(-1, (-1, 32))
    amax = groups.abs().amax(dim=-1).clamp_min(1e-12)
    exp = torch.ceil(torch.log2(amax / 6.0)).clamp_(-127.0, 127.0)
    scale_u8 = (exp + 127.0).to(torch.uint8)
    scaled = groups / torch.exp2(exp).unsqueeze(-1)
    levels = torch.tensor(_E2M1_LEVELS, device=x.device, dtype=torch.float32)
    mids = (levels[1:] + levels[:-1]) / 2.0
    mag = torch.bucketize(scaled.abs().contiguous(), mids)
    sign = (scaled < 0).to(torch.uint8)
    nibble = (sign << 3) | mag.to(torch.uint8)
    nibble = nibble.flatten(-2)
    packed = (nibble[..., 0::2] | (nibble[..., 1::2] << 4)).contiguous()
    return packed, scale_u8.contiguous()


def dequantize_mxfp4(
    packed: torch.Tensor, scales: torch.Tensor, dim: int, dtype=torch.float32
) -> torch.Tensor:
    """Torch reference dequant of :func:`quantize_mxfp4` output."""
    lo = packed & 0xF
    hi = packed >> 4
    nib = torch.stack((lo, hi), dim=-1).flatten(-2)[..., :dim]
    levels = torch.tensor(_E2M1_LEVELS, device=packed.device, dtype=torch.float32)
    mag = levels[(nib & 7).long()]
    val = torch.where(nib & 8 > 0, -mag, mag)
    exp = scales.float() - 127.0
    val = val.unflatten(-1, (-1, 32)) * torch.exp2(exp).unsqueeze(-1)
    return val.flatten(-2).to(dtype)


@triton.jit
def _e2m1_mag_to_f32(mag):
    # e2m1 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (3-bit codes 0..7)
    e = (mag >> 1) & 3
    m = (mag & 1).to(tl.float32)
    base = tl.where(e == 0, m * 0.5, tl.exp2(e.to(tl.float32) - 1.0) * (1.0 + m * 0.5))
    return base


@triton.jit
def _decode_paged_v2_kernel(
    q,
    k,
    v,
    k_scale,
    v_scale,
    q2k,
    page_table,
    seqused_k,
    out,
    lse_out,
    o_partial,
    lse_partial,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    seqlen_q: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    topk: tl.constexpr,
    split_pages: tl.constexpr,
    sparse: tl.constexpr,
    quantize_p_fp8: tl.constexpr,
    KV_FMT: tl.constexpr,  # 0 = fp8 unscaled, 1 = fp8 scaled, 2 = mxfp4
    WRITE_PARTIAL: tl.constexpr,
    dim: tl.constexpr,
):
    q_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    split_idx = tl.program_id(2)
    batch_idx = q_idx // seqlen_q
    q_local = q_idx - batch_idx * seqlen_q
    kv_head = q_head // qhead_per_kv
    slot_base = split_idx * split_pages

    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim
    q_vec = tl.load(q + (q_idx * head_q + q_head) * dim + offs_d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    used_k = tl.load(seqused_k + batch_idx)
    causal_limit = q_local + (used_k - seqlen_q)
    if KV_FMT == 1:
        v_ch = tl.load(v_scale + kv_head * dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((128,), tl.float32)

    for rel_slot in tl.static_range(0, split_pages):
        slot = slot_base + rel_slot
        logical_page = slot
        selected_valid = slot < topk
        if sparse:
            selected = tl.load(
                q2k + (kv_head * total_q + q_idx) * topk + slot,
                mask=(q_idx < total_q) & (slot < topk),
                other=-1,
            )
            logical_page = selected
            selected_valid = selected >= 0
        page_valid = selected_valid & (logical_page >= 0) & (logical_page < max_pages_per_seq)
        physical_page = tl.load(
            page_table + batch_idx * max_pages_per_seq + logical_page,
            mask=page_valid,
            other=0,
        )
        kv_pos = logical_page * blk_kv + offs_n
        token_valid = page_valid & (offs_n < blk_kv) & (kv_pos < used_k) & (kv_pos <= causal_limit)
        tok_base = (physical_page * head_kv + kv_head) * blk_kv + offs_n

        if KV_FMT == 2:
            offs_b = offs_d // 2
            k_byte = tl.load(
                k + (tok_base[None, :] * (dim // 2) + offs_b[:, None]),
                mask=token_valid[None, :] & d_mask[:, None],
                other=0,
            ).to(tl.uint8)
            use_hi = (offs_d & 1) != 0
            k_nib = tl.where(use_hi[:, None], k_byte >> 4, k_byte & 15)
            k_mag = _e2m1_mag_to_f32(k_nib & 7)
            k_val = tl.where((k_nib & 8) > 0, -k_mag, k_mag)
            k_sb = tl.load(
                k_scale + (tok_base[None, :] * (dim // 32) + (offs_d // 32)[:, None]),
                mask=token_valid[None, :] & d_mask[:, None],
                other=127,
            ).to(tl.float32)
            k_tile = k_val * tl.exp2(k_sb - 127.0)
        else:
            k_tile = tl.load(
                k + (tok_base[None, :] * dim + offs_d[:, None]),
                mask=token_valid[None, :] & d_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            if KV_FMT == 1:
                k_tok_scale = tl.load(k_scale + tok_base, mask=token_valid, other=0.0).to(
                    tl.float32
                )
                k_tile = k_tile * k_tok_scale[None, :]

        scores = tl.sum(k_tile * q_vec[:, None], axis=0) * softmax_scale
        scores = tl.where(token_valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(token_valid, p, 0.0)
        if quantize_p_fp8:
            # Scale into the e4m3 range first (p <= 1, e4m3 min normal ~2^-6):
            # a raw cast would flush most attention weights to zero.
            p_pv = (p * 448.0).to(tl.float8e4nv).to(tl.float32) * (1.0 / 448.0)
        else:
            p_pv = p

        if KV_FMT == 2:
            offs_b = offs_d // 2
            v_byte = tl.load(
                v + (tok_base[:, None] * (dim // 2) + offs_b[None, :]),
                mask=token_valid[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.uint8)
            use_hi_v = (offs_d & 1) != 0
            v_nib = tl.where(use_hi_v[None, :], v_byte >> 4, v_byte & 15)
            v_mag = _e2m1_mag_to_f32(v_nib & 7)
            v_val = tl.where((v_nib & 8) > 0, -v_mag, v_mag)
            v_sb = tl.load(
                v_scale + (tok_base[:, None] * (dim // 32) + (offs_d // 32)[None, :]),
                mask=token_valid[:, None] & d_mask[None, :],
                other=127,
            ).to(tl.float32)
            v_tile = v_val * tl.exp2(v_sb - 127.0)
        else:
            v_tile = tl.load(
                v + (tok_base[:, None] * dim + offs_d[None, :]),
                mask=token_valid[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if KV_FMT == 1:
                v_tile = v_tile * v_ch[None, :]

        acc = acc * alpha + tl.sum(p_pv[:, None] * v_tile, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    has_value = l_i > 0.0
    safe_l = tl.where(has_value, l_i, 1.0)
    out_vec = tl.where(has_value, acc / safe_l, 0.0)
    lse_val = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
    if WRITE_PARTIAL:
        tl.store(
            o_partial + ((split_idx * total_q + q_idx) * head_q + q_head) * dim + offs_d,
            out_vec,
            mask=d_mask,
        )
        tl.store(lse_partial + (split_idx * total_q + q_idx) * head_q + q_head, lse_val)
    else:
        tl.store(out + (q_idx * head_q + q_head) * dim + offs_d, out_vec, mask=d_mask)
        tl.store(lse_out + q_idx * head_q + q_head, lse_val)


def sparse_decode_paged_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: Optional[torch.Tensor],
    *,
    kv_format: str,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int,
    blk_kv: int,
    softmax_scale: Optional[float] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    split_pages: int = 0,
    return_softmax_lse: bool = False,
):
    """Paged sparse/dense decode with honest low-precision cache formats.

    kv_format:
      * ``fp8``        — e4m3 K/V, no scales (legacy contract)
      * ``fp8_scaled`` — e4m3 K/V + per-token K scale (pages, H, blk) and
                          per-channel V scale (H, dim)
      * ``mxfp4``      — packed e2m1 K/V (pages, H, blk, dim/2 u8) + ue8m0
                          scales (pages, H, blk, dim/32 u8)
    Q is e4m3 for fp8 formats and bf16/fp32 for mxfp4 (decode is
    bandwidth-bound; Q precision is irrelevant to the cache win).
    """
    fmt = {"fp8": 0, "fp8_scaled": 1, "mxfp4": 2}[kv_format]
    if fmt in (1, 2) and (k_scale is None or v_scale is None):
        raise ValueError(f"kv_format={kv_format} requires k_scale and v_scale")
    total_q, head_q, dim = q.shape
    if dim != 128:
        raise NotImplementedError("decode v2 supports D=128 only")
    if fmt == 2:
        head_kv = int(k.shape[1])
    else:
        head_kv = int(k.shape[1])
    max_pages_per_seq = int(page_table.shape[1])
    if q2k_indices is not None:
        q2k = q2k_indices.contiguous()
        topk = int(q2k.shape[-1])
        sparse = True
    else:
        q2k = page_table
        topk = max_pages_per_seq
        sparse = False
    quantize_p_fp8 = os.environ.get("FMHA_SM120_DECODE_FP8_P", "1").strip().lower() not in {
        "0",
        "false",
        "off",
    }
    out = torch.empty((total_q, head_q, dim), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    dummy = seqused_k
    ks_arg = k_scale if k_scale is not None else dummy
    vs_arg = v_scale if v_scale is not None else dummy

    if split_pages > 0 and topk > split_pages:
        num_splits = triton.cdiv(topk, split_pages)
        o_partial = torch.empty(
            (num_splits, total_q, head_q, dim), device=q.device, dtype=torch.float32
        )
        lse_partial = torch.empty(
            (num_splits, total_q, head_q), device=q.device, dtype=torch.float32
        )
        grid = (total_q, head_q, num_splits)
        _decode_paged_v2_kernel[grid](
            q, k, v, ks_arg, vs_arg, q2k, page_table, seqused_k,
            out, lse, o_partial, lse_partial,
            float(softmax_scale if softmax_scale is not None else dim**-0.5),
            int(total_q), int(head_q), int(head_kv), int(head_q // head_kv),
            int(seqlen_q), int(blk_kv), int(max_pages_per_seq), int(topk),
            int(split_pages), bool(sparse), bool(quantize_p_fp8),
            int(fmt), True, int(dim), num_warps=8,
        )
        from src.sm120.atten_triton import _sparse_decode_split_combine_kernel

        _sparse_decode_split_combine_kernel[(total_q, head_q)](
            o_partial, lse_partial, out, lse,
            int(total_q), int(head_q), int(num_splits), int(dim), num_warps=8,
        )
    else:
        grid = (total_q, head_q, 1)
        _decode_paged_v2_kernel[grid](
            q, k, v, ks_arg, vs_arg, q2k, page_table, seqused_k,
            out, lse, out, lse,
            float(softmax_scale if softmax_scale is not None else dim**-0.5),
            int(total_q), int(head_q), int(head_kv), int(head_q // head_kv),
            int(seqlen_q), int(blk_kv), int(max_pages_per_seq), int(topk),
            int(topk), bool(sparse), bool(quantize_p_fp8),
            int(fmt), False, int(dim), num_warps=8,
        )
    if return_softmax_lse:
        return out, lse
    return out


def quantize_paged_kv_fp8_scaled(
    k_pages: torch.Tensor, v_pages: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize paged bf16 K/V (pages, H, blk, D) to scaled e4m3 cache."""
    kf = k_pages.float()
    vf = v_pages.float()
    k_amax = kf.abs().amax(dim=-1).clamp_min(1e-6)  # (pages, H, blk)
    v_amax = vf.abs().amax(dim=(0, 2)).clamp_min(1e-6)  # (H, D)
    k_q = (kf * (FP8_MAX / k_amax).unsqueeze(-1)).to(torch.float8_e4m3fn)
    v_q = (vf * (FP8_MAX / v_amax).view(1, -1, 1, v_amax.shape[-1])).to(torch.float8_e4m3fn)
    return k_q, v_q, (k_amax / FP8_MAX).contiguous(), (v_amax / FP8_MAX).contiguous()


def quantize_paged_kv_mxfp4(
    k_pages: torch.Tensor, v_pages: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize paged bf16 K/V (pages, H, blk, D) to the MXFP4 cache format."""
    k_packed, k_scales = quantize_mxfp4(k_pages)
    v_packed, v_scales = quantize_mxfp4(v_pages)
    return k_packed, v_packed, k_scales, v_scales
