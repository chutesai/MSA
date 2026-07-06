# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Triton sparse attention backend for SM120 (RTX PRO 6000 Blackwell).

Implements the MSA CSR varlen contract without the SM100-only
CuTe/tcgen05/TMEM path: BF16 and FP8/NVFP4 K/V prefill (forward plus custom
autograd backward) and paged FP8 decode. q2k metadata is reconstructed from
the public CSR inputs when the caller does not pass it explicitly.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from src.sm120.reference import _reconstruct_q2k_from_k2q_csr, _to_int_list


_NVFP4_LUT_CACHE: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _nvfp4_luts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index if device.index is not None else -1)
    cached = _NVFP4_LUT_CACHE.get(key)
    if cached is not None:
        return cached

    fp4_vals = []
    for nibble in range(16):
        mag = nibble & 7
        value = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][mag]
        fp4_vals.append(-value if nibble & 8 else value)

    fp8_vals = []
    for byte in range(256):
        sign = (byte >> 7) & 1
        exp = (byte >> 3) & 15
        mant = byte & 7
        mant_f = mant * 0.125
        value = mant_f * math.exp2(-6.0) if exp == 0 else (1.0 + mant_f) * math.exp2(exp - 7.0)
        fp8_vals.append(-value if sign else value)

    cached = (
        torch.tensor(fp4_vals, device=device, dtype=torch.float32),
        torch.tensor(fp8_vals, device=device, dtype=torch.float32),
    )
    _NVFP4_LUT_CACHE[key] = cached
    return cached


@triton.jit
def _fp4_e2m1_to_f32(nibble):
    mag = nibble & 7
    val = tl.full(mag.shape, 0.0, tl.float32)
    val = tl.where(mag == 1, 0.5, val)
    val = tl.where(mag == 2, 1.0, val)
    val = tl.where(mag == 3, 1.5, val)
    val = tl.where(mag == 4, 2.0, val)
    val = tl.where(mag == 5, 3.0, val)
    val = tl.where(mag == 6, 4.0, val)
    val = tl.where(mag == 7, 6.0, val)
    return tl.where((nibble & 8) != 0, -val, val)


@triton.jit
def _fp8_e4m3fn_to_f32(byte):
    sign = (byte >> 7) & 1
    exp = (byte >> 3) & 15
    mant = byte & 7
    mant_f = mant.to(tl.float32) * 0.125
    sub = mant_f * tl.exp2(tl.full(mant.shape, -6.0, tl.float32))
    norm = (1.0 + mant_f) * tl.exp2(exp.to(tl.float32) - 7.0)
    val = tl.where(exp == 0, sub, norm)
    return tl.where(sign != 0, -val, val)


@triton.jit
def _scale_128x4_offset(row, col, scale_cols: tl.constexpr):
    tiles_n: tl.constexpr = (scale_cols + 3) // 4
    tile_m = row // 128
    tile_n = col // 4
    outer = row % 128
    inner = col % 4
    return (tile_m * tiles_n + tile_n) * 512 + (outer % 32) * 16 + (outer // 32) * 4 + inner


@triton.jit
def _sparse_attn_dense_bf16_kernel(
    q,
    k,
    v,
    q2k,
    page_table,
    cu_q,
    cu_k,
    out,
    lse_out,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_batch: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    dim: tl.constexpr,
):
    q_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // qhead_per_kv
    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim

    q_ptr = q + (q_idx * head_q + q_head) * dim + offs_d
    q_vec = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

    batch_idx = tl.full((), 0, tl.int32)
    q_local = q_idx
    q_len = tl.full((), 0, tl.int32)
    k_base = tl.full((), 0, tl.int32)
    k_len = tl.full((), 0, tl.int32)
    for b in tl.static_range(0, max_batch):
        q_start = tl.load(cu_q + b)
        q_end = tl.load(cu_q + b + 1)
        k_start = tl.load(cu_k + b)
        k_end = tl.load(cu_k + b + 1)
        in_batch = (q_idx >= q_start) & (q_idx < q_end)
        batch_idx = tl.where(in_batch, b, batch_idx)
        q_local = tl.where(in_batch, q_idx - q_start, q_local)
        q_len = tl.where(in_batch, q_end - q_start, q_len)
        k_base = tl.where(in_batch, k_start, k_base)
        k_len = tl.where(in_batch, k_end - k_start, k_len)

    acc = tl.zeros((128,), dtype=tl.float32)
    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)

    for slot in tl.static_range(0, topk):
        kv_block = tl.load(q2k + (kv_head * total_q + q_idx) * topk + slot)
        pos = kv_block * blk_kv + offs_n
        valid = (kv_block >= 0) & (offs_n < blk_kv) & (pos < k_len)
        if causal:
            valid = valid & (pos <= (q_local + (k_len - q_len)))
        if paged_kv:
            physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
            k_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        else:
            k_tok = k_base + pos
            k_ptrs = k + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
        k_tile = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(k_tile * q_vec[None, :], axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))

        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(valid, p, 0.0)

        if paged_kv:
            v_ptrs = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        else:
            v_ptrs = v + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
        v_tile = tl.load(v_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    has_value = l_i > 0.0
    out_vec = acc / l_i
    out_vec = tl.where(has_value, out_vec, 0.0)
    tl.store(out + (q_idx * head_q + q_head) * dim + offs_d, out_vec, mask=d_mask)
    tl.store(
        lse_out + q_idx * head_q + q_head,
        tl.where(has_value, tl.log(l_i) + m_i, -30000.0),  # finite empty-row sentinel
    )


@triton.jit
def _sparse_attn_dense_nvfp4_kernel(
    q,
    k,
    v,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
    q2k,
    page_table,
    cu_q,
    cu_k,
    out,
    lse_out,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_batch: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    has_k_global_scale: tl.constexpr,
    has_v_global_scale: tl.constexpr,
    dim: tl.constexpr,
):
    q_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // qhead_per_kv
    offs_d = tl.arange(0, 128)
    offs_b = offs_d // 2
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim
    scale_cols: tl.constexpr = 8

    q_vec = tl.load(q + (q_idx * head_q + q_head) * dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    kg = tl.load(k_global_scale) if has_k_global_scale else 1.0
    vg = tl.load(v_global_scale) if has_v_global_scale else 1.0

    batch_idx = tl.full((), 0, tl.int32)
    q_local = q_idx
    q_len = tl.full((), 0, tl.int32)
    k_base = tl.full((), 0, tl.int32)
    k_len = tl.full((), 0, tl.int32)
    for b in tl.static_range(0, max_batch):
        q_start = tl.load(cu_q + b)
        q_end = tl.load(cu_q + b + 1)
        k_start = tl.load(cu_k + b)
        k_end = tl.load(cu_k + b + 1)
        in_batch = (q_idx >= q_start) & (q_idx < q_end)
        batch_idx = tl.where(in_batch, b, batch_idx)
        q_local = tl.where(in_batch, q_idx - q_start, q_local)
        q_len = tl.where(in_batch, q_end - q_start, q_len)
        k_base = tl.where(in_batch, k_start, k_base)
        k_len = tl.where(in_batch, k_end - k_start, k_len)

    acc = tl.zeros((128,), dtype=tl.float32)
    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)

    for slot in tl.static_range(0, topk):
        kv_block = tl.load(q2k + (kv_head * total_q + q_idx) * topk + slot)
        safe_block = tl.maximum(kv_block, 0)
        pos = safe_block * blk_kv + offs_n
        valid = (kv_block >= 0) & (offs_n < blk_kv) & (pos < k_len)
        if causal:
            valid = valid & (pos <= (q_local + (k_len - q_len)))
        if paged_kv:
            physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + safe_block)
            k_byte_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * (dim // 2) + offs_b[None, :])
            v_byte_ptrs = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * (dim // 2) + offs_b[None, :])
            scale_row = (physical_page * head_kv + kv_head) * blk_kv + offs_n
        else:
            k_tok = k_base + pos
            k_byte_ptrs = k + ((k_tok[:, None] * head_kv + kv_head) * (dim // 2) + offs_b[None, :])
            v_byte_ptrs = v + ((k_tok[:, None] * head_kv + kv_head) * (dim // 2) + offs_b[None, :])
            scale_row = k_tok * head_kv + kv_head

        k_byte = tl.load(k_byte_ptrs, mask=valid[:, None] & d_mask[None, :], other=0)
        v_byte = tl.load(v_byte_ptrs, mask=valid[:, None] & d_mask[None, :], other=0)
        use_hi = (offs_d & 1) != 0
        k_nib = tl.where(use_hi[None, :], k_byte >> 4, k_byte & 15)
        v_nib = tl.where(use_hi[None, :], v_byte >> 4, v_byte & 15)
        scale_col = offs_d // 16
        scale_offsets = _scale_128x4_offset(scale_row[:, None], scale_col[None, :], scale_cols)
        k_scale_byte = tl.load(k_scale + scale_offsets, mask=valid[:, None] & d_mask[None, :], other=0)
        v_scale_byte = tl.load(v_scale + scale_offsets, mask=valid[:, None] & d_mask[None, :], other=0)
        k_tile = _fp4_e2m1_to_f32(k_nib) * _fp8_e4m3fn_to_f32(k_scale_byte) * kg
        scores = tl.sum(k_tile * q_vec[None, :], axis=1) * softmax_scale
        scores = tl.where(valid, scores, -float("inf"))

        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(valid, p, 0.0)
        v_tile = _fp4_e2m1_to_f32(v_nib) * _fp8_e4m3fn_to_f32(v_scale_byte) * vg
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    has_value = l_i > 0.0
    out_vec = acc / l_i
    out_vec = tl.where(has_value, out_vec, 0.0)
    tl.store(out + (q_idx * head_q + q_head) * dim + offs_d, out_vec, mask=d_mask)
    tl.store(
        lse_out + q_idx * head_q + q_head,
        tl.where(has_value, tl.log(l_i) + m_i, -30000.0),  # finite empty-row sentinel
    )


@triton.jit
def _nvfp4_dequant_to_bf16_kernel(
    src,
    scale,
    global_scale,
    fp4_lut,
    fp8_lut,
    dst,
    total_rows: tl.constexpr,
    has_global_scale: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < dim
    offs_b = offs_d // 2
    byte = tl.load(src + row * (dim // 2) + offs_b, mask=d_mask, other=0)
    use_hi = (offs_d & 1) != 0
    nib = tl.where(use_hi, byte >> 4, byte & 15)
    scale_col = offs_d // 16
    scale_offset = _scale_128x4_offset(row, scale_col, 8)
    scale_byte = tl.load(scale + scale_offset, mask=d_mask, other=0)
    gs = tl.load(global_scale) if has_global_scale else 1.0
    val = tl.load(fp4_lut + nib.to(tl.int32), mask=d_mask, other=0.0) * tl.load(
        fp8_lut + scale_byte.to(tl.int32), mask=d_mask, other=0.0
    ) * gs
    tl.store(dst + row * dim + offs_d, val, mask=d_mask)


@triton.jit
def _sparse_attn_csr_partial_bf16_kernel(
    q,
    k,
    v,
    q2k,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    o_partial,
    lse_partial,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    partial_q: tl.constexpr,
    q_start: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    has_qsplit: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_meta = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    qsplit_meta = tl.load(
        k2q_qsplit_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    if has_qsplit:
        q_local = qsplit_meta & 16777215
        split_slot = (qsplit_meta >> 24) & 255
    else:
        q_local = q_meta
        split_slot = tl.full((BLOCK_M,), -1, tl.int32)
    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)
    q_rel = q_global - q_start
    q_in_chunk = (q_rel >= 0) & (q_rel < partial_q)

    if not has_qsplit:
        for slot in tl.static_range(0, topk):
            selected = tl.load(
                q2k + (kv_head * total_q + q_global) * topk + slot,
                mask=q_valid,
                other=-999999,
            )
            split_slot = tl.where((split_slot < 0) & (selected == kv_block), slot, split_slot)
    q_valid = q_valid & q_in_chunk & (split_slot >= 0)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)

    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
    else:
        k_tok = k_base + pos
        k_ptrs = k + ((k_tok[None, :] * head_kv + kv_head) * dim + offs_d[:, None])
    k_tile = tl.load(k_ptrs, mask=kv_valid[None, :] & d_mask[:, None], other=0.0).to(tl.bfloat16)
    scores = tl.dot(q_tile.to(tl.bfloat16), k_tile, out_dtype=tl.float32) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    p = tl.exp(scores - row_max[:, None])
    p = tl.where(q_valid[:, None] & token_valid, p, 0.0)
    row_sum = tl.sum(p, axis=1)
    lse = tl.log(row_sum) + row_max
    has_value = q_valid & (row_sum > 0.0)

    if paged_kv:
        v_ptrs = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
    else:
        v_ptrs = v + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
    v_tile = tl.load(v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    o = tl.dot(p, v_tile, out_dtype=tl.float32)
    o = o / row_sum[:, None]
    o = tl.where(has_value[:, None], o, 0.0)

    o_ptrs = o_partial + ((split_slot[:, None] * partial_q + q_rel[:, None]) * head_q + q_head) * dim + offs_d[None, :]
    tl.store(o_ptrs, o, mask=has_value[:, None] & d_mask[None, :])
    tl.store(
        lse_partial + (split_slot * partial_q + q_rel) * head_q + q_head,
        tl.where(has_value, lse, -float("inf")),
        mask=q_valid,
    )


@triton.jit
def _sparse_attn_combine_kernel(
    o_partial,
    lse_partial,
    out,
    lse_out,
    total_q: tl.constexpr,
    partial_q: tl.constexpr,
    q_start: tl.constexpr,
    head_q: tl.constexpr,
    topk: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    q_block = tl.program_id(0)
    q_head = tl.program_id(1)
    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    q_global = q_start + offs_m
    q_valid = (offs_m < partial_q) & (q_global < total_q)
    d_mask = offs_d < dim

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, topk):
        lse = tl.load(
            lse_partial + (slot * partial_q + offs_m) * head_q + q_head,
            mask=q_valid,
            other=-float("inf"),
        )
        m = tl.maximum(m, lse)
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for slot in tl.static_range(0, topk):
        lse = tl.load(
            lse_partial + (slot * partial_q + offs_m) * head_q + q_head,
            mask=q_valid,
            other=-float("inf"),
        )
        w = tl.exp(lse - m)
        w = tl.where(lse > -float("inf"), w, 0.0)
        denom += w
        part = tl.load(
            o_partial + ((slot * partial_q + offs_m[:, None]) * head_q + q_head) * dim + offs_d[None, :],
            mask=q_valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += w[:, None] * part
    has_value = denom > 0.0
    out_val = acc / denom[:, None]
    out_val = tl.where(has_value[:, None], out_val, 0.0)
    tl.store(
        out + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :],
        out_val,
        mask=q_valid[:, None] & d_mask[None, :],
    )
    tl.store(
        lse_out + q_global * head_q + q_head,
        tl.where(has_value, tl.log(denom) + m, -30000.0),  # finite empty-row sentinel
        mask=q_valid,
    )


@triton.jit
def _sparse_attn_csr_lse_bf16_kernel(
    q,
    k,
    q2k,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    lse_partial,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    has_qsplit: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_meta = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    qsplit_meta = tl.load(
        k2q_qsplit_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    if has_qsplit:
        q_local = qsplit_meta & 16777215
        split_slot = (qsplit_meta >> 24) & 255
    else:
        q_local = q_meta
        split_slot = tl.full((BLOCK_M,), -1, tl.int32)
    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)

    if not has_qsplit:
        for slot in tl.static_range(0, topk):
            selected = tl.load(
                q2k + (kv_head * total_q + q_global) * topk + slot,
                mask=q_valid,
                other=-999999,
            )
            split_slot = tl.where((split_slot < 0) & (selected == kv_block), slot, split_slot)
    q_valid = q_valid & (split_slot >= 0)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
    else:
        k_tok = k_base + pos
        k_ptrs = k + ((k_tok[None, :] * head_kv + kv_head) * dim + offs_d[:, None])
    k_tile = tl.load(k_ptrs, mask=kv_valid[None, :] & d_mask[:, None], other=0.0).to(tl.bfloat16)
    scores = tl.dot(q_tile.to(tl.bfloat16), k_tile, out_dtype=tl.float32) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))
    row_max = tl.max(scores, axis=1)
    p = tl.exp(scores - row_max[:, None])
    p = tl.where(q_valid[:, None] & token_valid, p, 0.0)
    row_sum = tl.sum(p, axis=1)
    has_value = q_valid & (row_sum > 0.0)
    lse = tl.log(row_sum) + row_max
    tl.store(
        lse_partial + (split_slot * total_q + q_global) * head_q + q_head,
        tl.where(has_value, lse, -float("inf")),
        mask=q_valid,
    )


@triton.jit
def _sparse_attn_lse_combine_kernel(
    lse_partial,
    lse_out,
    total_q: tl.constexpr,
    head_q: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    q_block = tl.program_id(0)
    q_head = tl.program_id(1)
    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q_valid = offs_m < total_q
    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, topk):
        lse = tl.load(
            lse_partial + (slot * total_q + offs_m) * head_q + q_head,
            mask=q_valid,
            other=-float("inf"),
        )
        m = tl.maximum(m, lse)
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for slot in tl.static_range(0, topk):
        lse = tl.load(
            lse_partial + (slot * total_q + offs_m) * head_q + q_head,
            mask=q_valid,
            other=-float("inf"),
        )
        w = tl.exp(lse - m)
        w = tl.where(lse > -float("inf"), w, 0.0)
        denom += w
    has_value = denom > 0.0
    tl.store(
        lse_out + offs_m * head_q + q_head,
        tl.where(has_value, tl.log(denom) + m, -30000.0),  # finite empty-row sentinel
        mask=q_valid,
    )


@triton.jit
def _sparse_attn_csr_accum_bf16_kernel(
    q,
    k,
    v,
    q2k,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    lse_out,
    acc_out,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    has_qsplit: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_meta = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    qsplit_meta = tl.load(
        k2q_qsplit_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    if has_qsplit:
        q_local = qsplit_meta & 16777215
        split_slot = (qsplit_meta >> 24) & 255
    else:
        q_local = q_meta
        split_slot = tl.full((BLOCK_M,), -1, tl.int32)
    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)
    if not has_qsplit:
        for slot in tl.static_range(0, topk):
            selected = tl.load(
                q2k + (kv_head * total_q + q_global) * topk + slot,
                mask=q_valid,
                other=-999999,
            )
            split_slot = tl.where((split_slot < 0) & (selected == kv_block), slot, split_slot)
    q_valid = q_valid & (split_slot >= 0)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
        v_ptrs = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
    else:
        k_tok = k_base + pos
        k_ptrs = k + ((k_tok[None, :] * head_kv + kv_head) * dim + offs_d[:, None])
        v_ptrs = v + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
    k_tile = tl.load(k_ptrs, mask=kv_valid[None, :] & d_mask[:, None], other=0.0).to(tl.bfloat16)
    scores = tl.dot(q_tile.to(tl.bfloat16), k_tile, out_dtype=tl.float32) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))
    final_lse = tl.load(lse_out + q_global * head_q + q_head, mask=q_valid, other=-float("inf"))
    p = tl.exp(scores - final_lse[:, None])
    p = tl.where(q_valid[:, None] & token_valid & (final_lse > -1.0e4)[:, None], p, 0.0)  # sentinel-aware (empty rows)
    v_tile = tl.load(v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    contrib = tl.dot(p, v_tile, out_dtype=tl.float32)
    acc_ptrs = acc_out + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    tl.atomic_add(acc_ptrs, contrib, sem="relaxed", mask=q_valid[:, None] & d_mask[None, :])


@triton.jit
def _sparse_attn_csr_lse_nvfp4_kernel(
    q,
    k,
    k_scale,
    k_global_scale,
    q2k,
    k2q_row_ptr,
    k2q_q_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    lse_partial,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    has_k_global_scale: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_b = offs_d // 2
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim
    scale_cols: tl.constexpr = 8
    kg = tl.load(k_global_scale) if has_k_global_scale else 1.0

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_local = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)

    split_slot = tl.full((BLOCK_M,), -1, tl.int32)
    for slot in tl.static_range(0, topk):
        selected = tl.load(
            q2k + (kv_head * total_q + q_global) * topk + slot,
            mask=q_valid,
            other=-999999,
        )
        split_slot = tl.where((split_slot < 0) & (selected == kv_block), slot, split_slot)
    q_valid = q_valid & (split_slot >= 0)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0).to(tl.bfloat16)
    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_byte_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * (dim // 2) + offs_b[:, None])
        scale_row = (physical_page * head_kv + kv_head) * blk_kv + offs_n
    else:
        k_tok = k_base + pos
        k_byte_ptrs = k + ((k_tok[None, :] * head_kv + kv_head) * (dim // 2) + offs_b[:, None])
        scale_row = k_tok * head_kv + kv_head
    k_byte = tl.load(k_byte_ptrs, mask=d_mask[:, None] & kv_valid[None, :], other=0)
    use_hi = (offs_d & 1) != 0
    k_nib = tl.where(use_hi[:, None], k_byte >> 4, k_byte & 15)
    scale_col = offs_d // 16
    scale_offsets = _scale_128x4_offset(scale_row[None, :], scale_col[:, None], scale_cols)
    k_scale_byte = tl.load(k_scale + scale_offsets, mask=d_mask[:, None] & kv_valid[None, :], other=0)
    k_tile = (_fp4_e2m1_to_f32(k_nib) * _fp8_e4m3fn_to_f32(k_scale_byte) * kg).to(tl.bfloat16)
    scores = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))
    row_max = tl.max(scores, axis=1)
    p = tl.exp(scores - row_max[:, None])
    p = tl.where(q_valid[:, None] & token_valid, p, 0.0)
    row_sum = tl.sum(p, axis=1)
    has_value = q_valid & (row_sum > 0.0)
    lse = tl.log(row_sum) + row_max
    tl.store(
        lse_partial + (split_slot * total_q + q_global) * head_q + q_head,
        tl.where(has_value, lse, -float("inf")),
        mask=q_valid,
    )


@triton.jit
def _sparse_attn_csr_accum_nvfp4_kernel(
    q,
    k,
    v,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
    q2k,
    k2q_row_ptr,
    k2q_q_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    lse_out,
    acc_out,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    has_k_global_scale: tl.constexpr,
    has_v_global_scale: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_b = offs_d // 2
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim
    scale_cols: tl.constexpr = 8
    kg = tl.load(k_global_scale) if has_k_global_scale else 1.0
    vg = tl.load(v_global_scale) if has_v_global_scale else 1.0

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_local = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)

    split_slot = tl.full((BLOCK_M,), -1, tl.int32)
    for slot in tl.static_range(0, topk):
        selected = tl.load(
            q2k + (kv_head * total_q + q_global) * topk + slot,
            mask=q_valid,
            other=-999999,
        )
        split_slot = tl.where((split_slot < 0) & (selected == kv_block), slot, split_slot)
    q_valid = q_valid & (split_slot >= 0)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0).to(tl.bfloat16)
    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_byte_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * (dim // 2) + offs_b[:, None])
        v_byte_ptrs = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * (dim // 2) + offs_b[None, :])
        scale_row = (physical_page * head_kv + kv_head) * blk_kv + offs_n
    else:
        k_tok = k_base + pos
        k_byte_ptrs = k + ((k_tok[None, :] * head_kv + kv_head) * (dim // 2) + offs_b[:, None])
        v_byte_ptrs = v + ((k_tok[:, None] * head_kv + kv_head) * (dim // 2) + offs_b[None, :])
        scale_row = k_tok * head_kv + kv_head
    k_byte = tl.load(k_byte_ptrs, mask=d_mask[:, None] & kv_valid[None, :], other=0)
    use_hi_d = (offs_d & 1) != 0
    k_nib = tl.where(use_hi_d[:, None], k_byte >> 4, k_byte & 15)
    scale_col = offs_d // 16
    k_scale_offsets = _scale_128x4_offset(scale_row[None, :], scale_col[:, None], scale_cols)
    k_scale_byte = tl.load(k_scale + k_scale_offsets, mask=d_mask[:, None] & kv_valid[None, :], other=0)
    k_tile = (_fp4_e2m1_to_f32(k_nib) * _fp8_e4m3fn_to_f32(k_scale_byte) * kg).to(tl.bfloat16)
    scores = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))
    final_lse = tl.load(lse_out + q_global * head_q + q_head, mask=q_valid, other=-float("inf"))
    p = tl.exp(scores - final_lse[:, None])
    p = tl.where(q_valid[:, None] & token_valid & (final_lse > -1.0e4)[:, None], p, 0.0)  # sentinel-aware (empty rows)

    v_byte = tl.load(v_byte_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0)
    use_hi_v = (offs_d & 1) != 0
    v_nib = tl.where(use_hi_v[None, :], v_byte >> 4, v_byte & 15)
    v_scale_offsets = _scale_128x4_offset(scale_row[:, None], scale_col[None, :], scale_cols)
    v_scale_byte = tl.load(v_scale + v_scale_offsets, mask=kv_valid[:, None] & d_mask[None, :], other=0)
    v_tile = _fp4_e2m1_to_f32(v_nib) * _fp8_e4m3fn_to_f32(v_scale_byte) * vg
    contrib = tl.dot(p, v_tile, out_dtype=tl.float32)
    acc_ptrs = acc_out + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    tl.atomic_add(acc_ptrs, contrib, sem="relaxed", mask=q_valid[:, None] & d_mask[None, :])


@triton.jit
def _sparse_attn_csr_lse_fp8_kernel(
    q,
    k,
    q2k,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    lse_partial,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    has_qsplit: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_meta = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    qsplit_meta = tl.load(
        k2q_qsplit_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    if has_qsplit:
        q_local = qsplit_meta & 16777215
        split_slot = (qsplit_meta >> 24) & 255
    else:
        q_local = q_meta
        split_slot = tl.full((BLOCK_M,), -1, tl.int32)
    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)

    if not has_qsplit:
        for slot in tl.static_range(0, topk):
            selected = tl.load(
                q2k + (kv_head * total_q + q_global) * topk + slot,
                mask=q_valid,
                other=-999999,
            )
            split_slot = tl.where((split_slot < 0) & (selected == kv_block), slot, split_slot)
    q_valid = q_valid & (split_slot >= 0)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
    else:
        k_tok = k_base + pos
        k_ptrs = k + ((k_tok[None, :] * head_kv + kv_head) * dim + offs_d[:, None])
    k_tile = tl.load(k_ptrs, mask=kv_valid[None, :] & d_mask[:, None], other=0).to(tl.uint8)
    scores = tl.dot_scaled(
        q_tile.to(tl.bfloat16),
        None,
        "bf16",
        k_tile,
        None,
        "e4m3",
        out_dtype=tl.float32,
    ) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))
    row_max = tl.max(scores, axis=1)
    p = tl.exp(scores - row_max[:, None])
    p = tl.where(q_valid[:, None] & token_valid, p, 0.0)
    row_sum = tl.sum(p, axis=1)
    has_value = q_valid & (row_sum > 0.0)
    lse = tl.log(row_sum) + row_max
    tl.store(
        lse_partial + (split_slot * total_q + q_global) * head_q + q_head,
        tl.where(has_value, lse, -float("inf")),
        mask=q_valid,
    )


@triton.jit
def _sparse_attn_csr_accum_fp8_kernel(
    q,
    k,
    v,
    q2k,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    lse_out,
    acc_out,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    has_qsplit: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_meta = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    qsplit_meta = tl.load(
        k2q_qsplit_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    if has_qsplit:
        q_local = qsplit_meta & 16777215
        split_slot = (qsplit_meta >> 24) & 255
    else:
        q_local = q_meta
        split_slot = tl.full((BLOCK_M,), -1, tl.int32)
    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)
    if not has_qsplit:
        for slot in tl.static_range(0, topk):
            selected = tl.load(
                q2k + (kv_head * total_q + q_global) * topk + slot,
                mask=q_valid,
                other=-999999,
            )
            split_slot = tl.where((split_slot < 0) & (selected == kv_block), slot, split_slot)
    q_valid = q_valid & (split_slot >= 0)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
        v_ptrs = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
    else:
        k_tok = k_base + pos
        k_ptrs = k + ((k_tok[None, :] * head_kv + kv_head) * dim + offs_d[:, None])
        v_ptrs = v + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
    k_tile = tl.load(k_ptrs, mask=kv_valid[None, :] & d_mask[:, None], other=0).to(tl.uint8)
    scores = tl.dot_scaled(
        q_tile.to(tl.bfloat16),
        None,
        "bf16",
        k_tile,
        None,
        "e4m3",
        out_dtype=tl.float32,
    ) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))
    final_lse = tl.load(lse_out + q_global * head_q + q_head, mask=q_valid, other=-float("inf"))
    p = tl.exp(scores - final_lse[:, None])
    p = tl.where(q_valid[:, None] & token_valid & (final_lse > -1.0e4)[:, None], p, 0.0)  # sentinel-aware (empty rows)
    v_tile = tl.load(v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0).to(tl.uint8)
    contrib = tl.dot_scaled(
        p.to(tl.bfloat16),
        None,
        "bf16",
        v_tile,
        None,
        "e4m3",
        out_dtype=tl.float32,
    )
    acc_ptrs = acc_out + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    tl.atomic_add(acc_ptrs, contrib, sem="relaxed", mask=q_valid[:, None] & d_mask[None, :])


@triton.jit
def _sparse_attn_cast_acc_kernel(
    acc_out,
    out,
    total_q: tl.constexpr,
    head_q: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    q_block = tl.program_id(0)
    q_head = tl.program_id(1)
    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    q_valid = offs_m < total_q
    d_mask = offs_d < dim
    val = tl.load(
        acc_out + (offs_m[:, None] * head_q + q_head) * dim + offs_d[None, :],
        mask=q_valid[:, None] & d_mask[None, :],
        other=0.0,
    )
    tl.store(
        out + (offs_m[:, None] * head_q + q_head) * dim + offs_d[None, :],
        val,
        mask=q_valid[:, None] & d_mask[None, :],
    )


@triton.jit
def _sparse_attn_bwd_row_bf16_kernel(
    q,
    k,
    v,
    q2k,
    page_table,
    cu_q,
    cu_k,
    out,
    lse_out,
    dout,
    dq,
    dk,
    dv,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_batch: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    dim: tl.constexpr,
):
    q_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // qhead_per_kv
    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim

    batch_idx = tl.full((), 0, tl.int32)
    q_local = q_idx
    q_len = tl.full((), 0, tl.int32)
    k_base = tl.full((), 0, tl.int32)
    k_len = tl.full((), 0, tl.int32)
    for b in tl.static_range(0, max_batch):
        q_start = tl.load(cu_q + b)
        q_end = tl.load(cu_q + b + 1)
        k_start = tl.load(cu_k + b)
        k_end = tl.load(cu_k + b + 1)
        in_batch = (q_idx >= q_start) & (q_idx < q_end)
        batch_idx = tl.where(in_batch, b, batch_idx)
        q_local = tl.where(in_batch, q_idx - q_start, q_local)
        q_len = tl.where(in_batch, q_end - q_start, q_len)
        k_base = tl.where(in_batch, k_start, k_base)
        k_len = tl.where(in_batch, k_end - k_start, k_len)

    q_vec = tl.load(q + (q_idx * head_q + q_head) * dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    do_vec = tl.load(dout + (q_idx * head_q + q_head) * dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    out_vec = tl.load(out + (q_idx * head_q + q_head) * dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    lse = tl.load(lse_out + q_idx * head_q + q_head)
    has_row = lse > -1.0e4  # sentinel-aware
    do_dot_o = tl.sum(do_vec * out_vec, axis=0)
    dq_acc = tl.zeros((128,), dtype=tl.float32)

    for slot in tl.static_range(0, topk):
        kv_block = tl.load(q2k + (kv_head * total_q + q_idx) * topk + slot)
        safe_block = tl.maximum(kv_block, 0)
        pos = safe_block * blk_kv + offs_n
        valid = (kv_block >= 0) & (offs_n < blk_kv) & (pos < k_len)
        if causal:
            valid = valid & (pos <= (q_local + (k_len - q_len)))

        if paged_kv:
            physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + safe_block)
            k_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
            v_ptrs = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
            dk_ptrs = dk + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
            dv_ptrs = dv + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        else:
            k_tok = k_base + pos
            k_ptrs = k + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
            v_ptrs = v + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
            dk_ptrs = dk + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
            dv_ptrs = dv + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])

        k_tile = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        logits = tl.sum(k_tile * q_vec[None, :], axis=1) * softmax_scale
        p = tl.exp(logits - lse)
        p = tl.where(valid & has_row, p, 0.0)
        dp = tl.sum(v_tile * do_vec[None, :], axis=1)
        ds = p * (dp - do_dot_o)
        dq_acc += tl.sum((ds[:, None] * k_tile), axis=0) * softmax_scale
        dk_update = (ds[:, None] * q_vec[None, :]) * softmax_scale
        dv_update = p[:, None] * do_vec[None, :]
        tl.atomic_add(dk_ptrs, dk_update, sem="relaxed", mask=valid[:, None] & d_mask[None, :])
        tl.atomic_add(dv_ptrs, dv_update, sem="relaxed", mask=valid[:, None] & d_mask[None, :])

    tl.store(dq + (q_idx * head_q + q_head) * dim + offs_d, dq_acc, mask=d_mask)


@triton.jit
def _sparse_attn_bwd_csr_bf16_kernel(
    q,
    k,
    v,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    out,
    lse_out,
    dout,
    dq,
    dk,
    dv,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    has_qsplit: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_meta = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    qsplit_meta = tl.load(
        k2q_qsplit_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    if has_qsplit:
        q_local = qsplit_meta & 16777215
    else:
        q_local = q_meta

    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    do_ptrs = dout + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    out_ptrs = out + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    do_tile = tl.load(do_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    out_tile = tl.load(out_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)

    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_ptrs_qk = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
        v_ptrs_dp = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
        k_ptrs_grad = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        dk_ptrs = dk + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        dv_ptrs = dv + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
    else:
        k_tok = k_base + pos
        k_ptrs_qk = k + ((k_tok[None, :] * head_kv + kv_head) * dim + offs_d[:, None])
        v_ptrs_dp = v + ((k_tok[None, :] * head_kv + kv_head) * dim + offs_d[:, None])
        k_ptrs_grad = k + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
        dk_ptrs = dk + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
        dv_ptrs = dv + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])

    k_qk = tl.load(k_ptrs_qk, mask=kv_valid[None, :] & d_mask[:, None], other=0.0)
    scores = tl.dot(q_tile, k_qk, out_dtype=tl.float32) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))

    lse = tl.load(lse_out + q_global * head_q + q_head, mask=q_valid, other=-float("inf"))
    p = tl.exp(scores - lse[:, None])
    p = tl.where(q_valid[:, None] & token_valid & (lse > -1.0e4)[:, None], p, 0.0)  # sentinel-aware (empty rows)

    v_dp = tl.load(v_ptrs_dp, mask=kv_valid[None, :] & d_mask[:, None], other=0.0)
    dp = tl.dot(do_tile, v_dp, out_dtype=tl.float32)
    do_dot_o = tl.sum(do_tile * out_tile, axis=1)
    ds = p * (dp - do_dot_o[:, None])

    k_grad = tl.load(k_ptrs_grad, mask=kv_valid[:, None] & d_mask[None, :], other=0.0)
    ds_mma = ds.to(tl.bfloat16)
    p_mma = p.to(tl.bfloat16)
    dq_update = tl.dot(ds_mma, k_grad, out_dtype=tl.float32) * softmax_scale
    dq_ptrs = dq + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    tl.atomic_add(dq_ptrs, dq_update, sem="relaxed", mask=q_valid[:, None] & d_mask[None, :])

    dk_update = tl.dot(tl.trans(ds_mma), q_tile, out_dtype=tl.float32) * softmax_scale
    dv_update = tl.dot(tl.trans(p_mma), do_tile, out_dtype=tl.float32)
    tl.atomic_add(dk_ptrs, dk_update, sem="relaxed", mask=kv_valid[:, None] & d_mask[None, :])
    tl.atomic_add(dv_ptrs, dv_update, sem="relaxed", mask=kv_valid[:, None] & d_mask[None, :])


@triton.jit
def _sparse_attn_bwd_csr_fp8_kernel(
    q,
    k,
    v,
    k2q_row_ptr,
    k2q_q_indices,
    k2q_qsplit_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    out,
    lse_out,
    dout,
    dq,
    dk,
    dv,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    has_qsplit: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_meta = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    qsplit_meta = tl.load(
        k2q_qsplit_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    if has_qsplit:
        q_local = qsplit_meta & 16777215
    else:
        q_local = q_meta

    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    do_ptrs = dout + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    out_ptrs = out + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    do_tile = tl.load(do_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    out_tile = tl.load(out_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)

    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_ptrs_qk = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
        v_ptrs_dp = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
        k_ptrs_grad = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        dk_ptrs = dk + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        dv_ptrs = dv + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
    else:
        k_tok = k_base + pos
        k_ptrs_qk = k + ((k_tok[None, :] * head_kv + kv_head) * dim + offs_d[:, None])
        v_ptrs_dp = v + ((k_tok[None, :] * head_kv + kv_head) * dim + offs_d[:, None])
        k_ptrs_grad = k + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
        dk_ptrs = dk + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
        dv_ptrs = dv + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])

    k_qk = tl.load(k_ptrs_qk, mask=kv_valid[None, :] & d_mask[:, None], other=0).to(tl.uint8)
    scores = tl.dot_scaled(
        q_tile.to(tl.bfloat16),
        None,
        "bf16",
        k_qk,
        None,
        "e4m3",
        out_dtype=tl.float32,
    ) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))

    lse = tl.load(lse_out + q_global * head_q + q_head, mask=q_valid, other=-float("inf"))
    p = tl.exp(scores - lse[:, None])
    p = tl.where(q_valid[:, None] & token_valid & (lse > -1.0e4)[:, None], p, 0.0)  # sentinel-aware (empty rows)

    v_dp = tl.load(v_ptrs_dp, mask=kv_valid[None, :] & d_mask[:, None], other=0).to(tl.uint8)
    dp = tl.dot_scaled(
        do_tile.to(tl.bfloat16),
        None,
        "bf16",
        v_dp,
        None,
        "e4m3",
        out_dtype=tl.float32,
    )
    do_dot_o = tl.sum(do_tile * out_tile, axis=1)
    ds = p * (dp - do_dot_o[:, None])

    k_grad = tl.load(k_ptrs_grad, mask=kv_valid[:, None] & d_mask[None, :], other=0).to(tl.uint8)
    ds_mma = ds.to(tl.bfloat16)
    p_mma = p.to(tl.bfloat16)
    dq_update = tl.dot_scaled(
        ds_mma,
        None,
        "bf16",
        k_grad,
        None,
        "e4m3",
        out_dtype=tl.float32,
    ) * softmax_scale
    dq_ptrs = dq + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    tl.atomic_add(dq_ptrs, dq_update, sem="relaxed", mask=q_valid[:, None] & d_mask[None, :])

    dk_update = tl.dot(tl.trans(ds_mma), q_tile, out_dtype=tl.float32) * softmax_scale
    dv_update = tl.dot(tl.trans(p_mma), do_tile, out_dtype=tl.float32)
    tl.atomic_add(dk_ptrs, dk_update, sem="relaxed", mask=kv_valid[:, None] & d_mask[None, :])
    tl.atomic_add(dv_ptrs, dv_update, sem="relaxed", mask=kv_valid[:, None] & d_mask[None, :])


@triton.jit
def _sparse_attn_bwd_csr_nvfp4_kernel(
    q,
    k,
    v,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
    q2k,
    k2q_row_ptr,
    k2q_q_indices,
    row_batch,
    row_kv_block,
    page_table,
    cu_q,
    cu_k,
    out,
    lse_out,
    dout,
    dq,
    dk,
    dv,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    total_rows: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    topk: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    num_row_chunks: tl.constexpr,
    causal: tl.constexpr,
    paged_kv: tl.constexpr,
    has_k_global_scale: tl.constexpr,
    has_v_global_scale: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    q_rep = tl.program_id(1)
    chunk = pid % num_row_chunks
    row_h = pid // num_row_chunks
    row = row_h % total_rows
    kv_head = row_h // total_rows
    q_head = kv_head * qhead_per_kv + q_rep

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    offs_b = offs_d // 2
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim
    scale_cols: tl.constexpr = 8
    kg = tl.load(k_global_scale) if has_k_global_scale else 1.0
    vg = tl.load(v_global_scale) if has_v_global_scale else 1.0

    row_start = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row)
    row_end = tl.load(k2q_row_ptr + kv_head * (total_rows + 1) + row + 1)
    csr_offs = row_start + chunk * BLOCK_M + offs_m
    q_local = tl.load(
        k2q_q_indices + kv_head * (total_q * topk) + csr_offs,
        mask=csr_offs < row_end,
        other=-1,
    )
    batch_idx = tl.load(row_batch + row)
    kv_block = tl.load(row_kv_block + row)
    q_base = tl.load(cu_q + batch_idx)
    q_next = tl.load(cu_q + batch_idx + 1)
    k_base = tl.load(cu_k + batch_idx)
    k_next = tl.load(cu_k + batch_idx + 1)
    q_len = q_next - q_base
    k_len = k_next - k_base
    q_global = q_base + q_local
    q_valid = (csr_offs < row_end) & (q_local >= 0) & (q_local < q_len)

    split_slot = tl.full((BLOCK_M,), -1, tl.int32)
    for slot in tl.static_range(0, topk):
        selected = tl.load(
            q2k + (kv_head * total_q + q_global) * topk + slot,
            mask=q_valid,
            other=-999999,
        )
        split_slot = tl.where((split_slot < 0) & (selected == kv_block), slot, split_slot)
    q_valid = q_valid & (split_slot >= 0)

    q_ptrs = q + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    do_ptrs = dout + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    out_ptrs = out + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    do_tile = tl.load(do_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)
    out_tile = tl.load(out_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)

    pos = kv_block * blk_kv + offs_n
    kv_valid = (offs_n < blk_kv) & (pos < k_len)
    if paged_kv:
        physical_page = tl.load(page_table + batch_idx * max_pages_per_seq + kv_block)
        k_byte_ptrs_qk = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * (dim // 2) + offs_b[:, None])
        k_byte_ptrs_grad = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * (dim // 2) + offs_b[None, :])
        v_byte_ptrs_dp = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * (dim // 2) + offs_b[:, None])
        dk_ptrs = dk + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        dv_ptrs = dv + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        scale_row = (physical_page * head_kv + kv_head) * blk_kv + offs_n
    else:
        k_tok = k_base + pos
        k_byte_ptrs_qk = k + ((k_tok[None, :] * head_kv + kv_head) * (dim // 2) + offs_b[:, None])
        k_byte_ptrs_grad = k + ((k_tok[:, None] * head_kv + kv_head) * (dim // 2) + offs_b[None, :])
        v_byte_ptrs_dp = v + ((k_tok[None, :] * head_kv + kv_head) * (dim // 2) + offs_b[:, None])
        dk_ptrs = dk + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
        dv_ptrs = dv + ((k_tok[:, None] * head_kv + kv_head) * dim + offs_d[None, :])
        scale_row = k_tok * head_kv + kv_head

    use_hi = (offs_d & 1) != 0
    scale_col = offs_d // 16

    k_byte_qk = tl.load(k_byte_ptrs_qk, mask=d_mask[:, None] & kv_valid[None, :], other=0)
    k_nib_qk = tl.where(use_hi[:, None], k_byte_qk >> 4, k_byte_qk & 15)
    k_scale_offsets_qk = _scale_128x4_offset(scale_row[None, :], scale_col[:, None], scale_cols)
    k_scale_byte_qk = tl.load(k_scale + k_scale_offsets_qk, mask=d_mask[:, None] & kv_valid[None, :], other=0)
    k_qk = (_fp4_e2m1_to_f32(k_nib_qk) * _fp8_e4m3fn_to_f32(k_scale_byte_qk) * kg).to(tl.bfloat16)
    scores = tl.dot(q_tile.to(tl.bfloat16), k_qk, out_dtype=tl.float32) * softmax_scale
    token_valid = kv_valid[None, :]
    if causal:
        causal_limit = q_local + (k_len - q_len)
        token_valid = token_valid & (pos[None, :] <= causal_limit[:, None])
    scores = tl.where(q_valid[:, None] & token_valid, scores, -float("inf"))

    final_lse = tl.load(lse_out + q_global * head_q + q_head, mask=q_valid, other=-float("inf"))
    p = tl.exp(scores - final_lse[:, None])
    p = tl.where(q_valid[:, None] & token_valid & (final_lse > -1.0e4)[:, None], p, 0.0)  # sentinel-aware (empty rows)

    v_byte_dp = tl.load(v_byte_ptrs_dp, mask=d_mask[:, None] & kv_valid[None, :], other=0)
    v_nib_dp = tl.where(use_hi[:, None], v_byte_dp >> 4, v_byte_dp & 15)
    v_scale_offsets_dp = _scale_128x4_offset(scale_row[None, :], scale_col[:, None], scale_cols)
    v_scale_byte_dp = tl.load(v_scale + v_scale_offsets_dp, mask=d_mask[:, None] & kv_valid[None, :], other=0)
    v_dp = (_fp4_e2m1_to_f32(v_nib_dp) * _fp8_e4m3fn_to_f32(v_scale_byte_dp) * vg).to(tl.bfloat16)
    dp = tl.dot(do_tile, v_dp, out_dtype=tl.float32)
    do_dot_o = tl.sum(do_tile * out_tile, axis=1)
    ds = p * (dp - do_dot_o[:, None])

    k_byte_grad = tl.load(k_byte_ptrs_grad, mask=kv_valid[:, None] & d_mask[None, :], other=0)
    k_nib_grad = tl.where(use_hi[None, :], k_byte_grad >> 4, k_byte_grad & 15)
    k_scale_offsets_grad = _scale_128x4_offset(scale_row[:, None], scale_col[None, :], scale_cols)
    k_scale_byte_grad = tl.load(k_scale + k_scale_offsets_grad, mask=kv_valid[:, None] & d_mask[None, :], other=0)
    k_grad = (_fp4_e2m1_to_f32(k_nib_grad) * _fp8_e4m3fn_to_f32(k_scale_byte_grad) * kg).to(tl.bfloat16)

    ds_mma = ds.to(tl.bfloat16)
    p_mma = p.to(tl.bfloat16)
    dq_update = tl.dot(ds_mma, k_grad, out_dtype=tl.float32) * softmax_scale
    dq_ptrs = dq + (q_global[:, None] * head_q + q_head) * dim + offs_d[None, :]
    tl.atomic_add(dq_ptrs, dq_update, sem="relaxed", mask=q_valid[:, None] & d_mask[None, :])

    dk_update = tl.dot(tl.trans(ds_mma), q_tile, out_dtype=tl.float32) * softmax_scale
    dv_update = tl.dot(tl.trans(p_mma), do_tile, out_dtype=tl.float32)
    tl.atomic_add(dk_ptrs, dk_update, sem="relaxed", mask=kv_valid[:, None] & d_mask[None, :])
    tl.atomic_add(dv_ptrs, dv_update, sem="relaxed", mask=kv_valid[:, None] & d_mask[None, :])


def _build_row_maps(cu_seqlens_k: torch.Tensor, blk_kv: int) -> tuple[torch.Tensor, torch.Tensor]:
    cu_k = _to_int_list(cu_seqlens_k)
    row_batch: list[int] = []
    row_kv_block: list[int] = []
    rows_per_batch = [
        (max(cu_k[i + 1] - cu_k[i], 0) + blk_kv - 1) // blk_kv
        for i in range(len(cu_k) - 1)
    ]
    max_rows = max(rows_per_batch, default=0)
    for kv_block in range(max_rows):
        for batch_idx, row_count in enumerate(rows_per_batch):
            if kv_block < row_count:
                row_batch.append(batch_idx)
                row_kv_block.append(kv_block)
    device = cu_seqlens_k.device
    return (
        torch.tensor(row_batch, dtype=torch.int32, device=device),
        torch.tensor(row_kv_block, dtype=torch.int32, device=device),
    )


def _partial_dtype() -> torch.dtype:
    env = os.environ.get("FMHA_SM120_PARTIAL_DTYPE", "bf16").strip().lower()
    if env in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if env in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"FMHA_SM120_PARTIAL_DTYPE must be bf16 or fp32, got {env!r}")


def _forward_mode() -> str:
    mode = os.environ.get("FMHA_SM120_TRITON_MODE", "auto").strip().lower()
    if mode not in {"auto", "two_phase", "chunked", "recompute", "row", "qstat"}:
        raise ValueError(
            "FMHA_SM120_TRITON_MODE must be one of auto/two_phase/chunked/recompute/row/qstat, "
            f"got {mode!r}"
        )
    return mode


def _should_recompute(mode: str, total_q: int, head_q: int, dim: int, topk: int) -> bool:
    """In auto mode, use the recompute path when the two-phase O/LSE partials
    would exceed FMHA_SM120_MAX_PARTIAL_MIB."""
    if mode == "recompute":
        return True
    if mode != "auto":
        return False
    max_partial_mib = int(os.environ.get("FMHA_SM120_MAX_PARTIAL_MIB", "1024"))
    if max_partial_mib <= 0:
        return False
    partial_bytes = topk * total_q * head_q * (dim * _partial_dtype().itemsize + 4)
    return partial_bytes > max_partial_mib * 1024 * 1024


def _csr_block_m(default: int) -> int:
    """CSR rows per program, FMHA_SM120_BLOCK_M overriding a per-path default.

    Measured on RTX PRO 6000 (seq 8192, topk 16): 32 is fastest for the
    forward kernels, 64 for the atomics-heavy backwards.
    """
    return int(os.environ.get("FMHA_SM120_BLOCK_M", str(default)))


def _csr_launch_params(
    k2q_row_ptr: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    blk_kv: int,
    block_m: int,
) -> tuple[int, torch.Tensor, torch.Tensor, int]:
    """Grid geometry shared by the CSR-row-parallel kernels."""
    total_rows = int(k2q_row_ptr.shape[1] - 1)
    row_batch, row_kv_block = _build_row_maps(cu_seqlens_k, blk_kv)
    if int(row_batch.numel()) != total_rows:
        raise ValueError("row map size does not match k2q_row_ptr")
    row_counts = k2q_row_ptr[:, 1:] - k2q_row_ptr[:, :-1]
    max_row_count = int(row_counts.max().item()) if row_counts.numel() else 0
    num_row_chunks = triton.cdiv(max(max_row_count, 1), block_m)
    return total_rows, row_batch, row_kv_block, num_row_chunks


def _qsplit_arg(
    k2q_q_indices: torch.Tensor, k2q_qsplit_indices: Optional[torch.Tensor]
) -> tuple[torch.Tensor, bool]:
    """The packed qsplit CSR stream, or the plain q-index stream as a placeholder."""
    if k2q_qsplit_indices is None:
        return k2q_q_indices, False
    if k2q_qsplit_indices.shape != k2q_q_indices.shape:
        raise ValueError("k2q_qsplit_indices must match k2q_q_indices shape")
    return k2q_qsplit_indices.contiguous(), True


def _sparse_attention_csr_varlen_triton_two_phase(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    k2q_qsplit_indices: Optional[torch.Tensor],
    *,
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    return_softmax_lse: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    total_q, head_q, dim = q.shape
    head_kv = k.shape[1]
    qhead_per_kv = head_q // head_kv
    block_m = _csr_block_m(32)
    total_rows, row_batch, row_kv_block, num_row_chunks = _csr_launch_params(
        k2q_row_ptr, cu_seqlens_k, blk_kv, block_m
    )
    partial_dtype = _partial_dtype()
    partial_q = int(total_q)
    o_partial = torch.zeros(
        (int(topk), int(partial_q), int(head_q), int(dim)),
        device=q.device,
        dtype=partial_dtype,
    )
    lse_partial = torch.full(
        (int(topk), int(partial_q), int(head_q)),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    paged_kv = page_table is not None
    page_table_arg = page_table if paged_kv else cu_seqlens_q
    max_pages_per_seq = int(page_table.shape[1]) if paged_kv else 1
    qsplit_arg, has_qsplit = _qsplit_arg(k2q_q_indices, k2q_qsplit_indices)
    grid_partial = (int(head_kv) * int(total_rows) * int(num_row_chunks), int(qhead_per_kv))
    _sparse_attn_csr_partial_bf16_kernel[grid_partial](
        q,
        k,
        v,
        q2k,
        k2q_row_ptr,
        k2q_q_indices,
        qsplit_arg,
        row_batch,
        row_kv_block,
        page_table_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        o_partial,
        lse_partial,
        float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
        int(total_q),
        int(total_rows),
        int(partial_q),
        0,
        int(head_q),
        int(head_kv),
        int(qhead_per_kv),
        int(topk),
        int(blk_kv),
        int(max_pages_per_seq),
        int(num_row_chunks),
        bool(has_qsplit),
        bool(causal),
        bool(paged_kv),
        int(dim),
        BLOCK_M=block_m,
        num_warps=8,
    )
    out = torch.empty((total_q, head_q, dim), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    grid_combine = (triton.cdiv(int(total_q), block_m), int(head_q))
    _sparse_attn_combine_kernel[grid_combine](
        o_partial,
        lse_partial,
        out,
        lse,
        int(total_q),
        int(partial_q),
        0,
        int(head_q),
        int(topk),
        int(dim),
        BLOCK_M=block_m,
        num_warps=8,
    )
    if return_softmax_lse:
        return out, lse
    return out


def _sparse_attention_csr_varlen_triton_two_phase_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    k2q_qsplit_indices: Optional[torch.Tensor],
    *,
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    return_softmax_lse: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
    q_chunk: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    total_q, head_q, dim = q.shape
    head_kv = k.shape[1]
    qhead_per_kv = head_q // head_kv
    block_m = _csr_block_m(32)
    total_rows, row_batch, row_kv_block, num_row_chunks = _csr_launch_params(
        k2q_row_ptr, cu_seqlens_k, blk_kv, block_m
    )
    q_chunk = max(block_m, int(q_chunk))
    q_chunk = triton.cdiv(q_chunk, block_m) * block_m
    partial_dtype = _partial_dtype()
    paged_kv = page_table is not None
    page_table_arg = page_table if paged_kv else cu_seqlens_q
    max_pages_per_seq = int(page_table.shape[1]) if paged_kv else 1
    qsplit_arg, has_qsplit = _qsplit_arg(k2q_q_indices, k2q_qsplit_indices)

    out = torch.empty((total_q, head_q, dim), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    for q_start in range(0, int(total_q), int(q_chunk)):
        partial_q = min(int(q_chunk), int(total_q) - int(q_start))
        partial_q_padded = triton.cdiv(partial_q, block_m) * block_m
        o_partial = torch.zeros(
            (int(topk), int(partial_q_padded), int(head_q), int(dim)),
            device=q.device,
            dtype=partial_dtype,
        )
        lse_partial = torch.full(
            (int(topk), int(partial_q_padded), int(head_q)),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        grid_partial = (int(head_kv) * int(total_rows) * int(num_row_chunks), int(qhead_per_kv))
        _sparse_attn_csr_partial_bf16_kernel[grid_partial](
            q,
            k,
            v,
            q2k,
            k2q_row_ptr,
            k2q_q_indices,
            qsplit_arg,
            row_batch,
            row_kv_block,
            page_table_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            o_partial,
            lse_partial,
            float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
            int(total_q),
            int(total_rows),
            int(partial_q_padded),
            int(q_start),
            int(head_q),
            int(head_kv),
            int(qhead_per_kv),
            int(topk),
            int(blk_kv),
            int(max_pages_per_seq),
            int(num_row_chunks),
            bool(has_qsplit),
            bool(causal),
            bool(paged_kv),
            int(dim),
            BLOCK_M=block_m,
            num_warps=8,
        )
        grid_combine = (triton.cdiv(int(partial_q_padded), block_m), int(head_q))
        _sparse_attn_combine_kernel[grid_combine](
            o_partial,
            lse_partial,
            out,
            lse,
            int(total_q),
            int(partial_q_padded),
            int(q_start),
            int(head_q),
            int(topk),
            int(dim),
            BLOCK_M=block_m,
            num_warps=8,
        )
    if return_softmax_lse:
        return out, lse
    return out


def _sparse_attention_csr_varlen_triton_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    k2q_qsplit_indices: Optional[torch.Tensor],
    *,
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    return_softmax_lse: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    total_q, head_q, dim = q.shape
    head_kv = k.shape[1]
    qhead_per_kv = head_q // head_kv
    block_m = _csr_block_m(32)
    total_rows, row_batch, row_kv_block, num_row_chunks = _csr_launch_params(
        k2q_row_ptr, cu_seqlens_k, blk_kv, block_m
    )
    paged_kv = page_table is not None
    page_table_arg = page_table if paged_kv else cu_seqlens_q
    max_pages_per_seq = int(page_table.shape[1]) if paged_kv else 1
    qsplit_arg, has_qsplit = _qsplit_arg(k2q_q_indices, k2q_qsplit_indices)

    lse_partial = torch.full(
        (int(topk), int(total_q), int(head_q)),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    grid_partial = (int(head_kv) * int(total_rows) * int(num_row_chunks), int(qhead_per_kv))
    _sparse_attn_csr_lse_bf16_kernel[grid_partial](
        q,
        k,
        q2k,
        k2q_row_ptr,
        k2q_q_indices,
        qsplit_arg,
        row_batch,
        row_kv_block,
        page_table_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        lse_partial,
        float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
        int(total_q),
        int(total_rows),
        int(head_q),
        int(head_kv),
        int(qhead_per_kv),
        int(topk),
        int(blk_kv),
        int(max_pages_per_seq),
        int(num_row_chunks),
        bool(has_qsplit),
        bool(causal),
        bool(paged_kv),
        int(dim),
        BLOCK_M=block_m,
        num_warps=8,
    )
    grid_combine = (triton.cdiv(int(total_q), block_m), int(head_q))
    _sparse_attn_lse_combine_kernel[grid_combine](
        lse_partial,
        lse,
        int(total_q),
        int(head_q),
        int(topk),
        BLOCK_M=block_m,
        num_warps=8,
    )
    acc_out = torch.zeros((total_q, head_q, dim), device=q.device, dtype=torch.float32)
    _sparse_attn_csr_accum_bf16_kernel[grid_partial](
        q,
        k,
        v,
        q2k,
        k2q_row_ptr,
        k2q_q_indices,
        qsplit_arg,
        row_batch,
        row_kv_block,
        page_table_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        lse,
        acc_out,
        float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
        int(total_q),
        int(total_rows),
        int(head_q),
        int(head_kv),
        int(qhead_per_kv),
        int(topk),
        int(blk_kv),
        int(max_pages_per_seq),
        int(num_row_chunks),
        bool(has_qsplit),
        bool(causal),
        bool(paged_kv),
        int(dim),
        BLOCK_M=block_m,
        num_warps=8,
    )
    out = torch.empty((total_q, head_q, dim), device=q.device, dtype=torch.bfloat16)
    _sparse_attn_cast_acc_kernel[grid_combine](
        acc_out,
        out,
        int(total_q),
        int(head_q),
        int(dim),
        BLOCK_M=block_m,
        num_warps=8,
    )
    if return_softmax_lse:
        return out, lse
    return out


def _sparse_attention_csr_varlen_triton_recompute_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    k2q_qsplit_indices: Optional[torch.Tensor],
    *,
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    return_softmax_lse: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    total_q, head_q, dim = q.shape
    head_kv = k.shape[1]
    qhead_per_kv = head_q // head_kv
    block_m = _csr_block_m(32)
    total_rows, row_batch, row_kv_block, num_row_chunks = _csr_launch_params(
        k2q_row_ptr, cu_seqlens_k, blk_kv, block_m
    )
    paged_kv = page_table is not None
    page_table_arg = page_table if paged_kv else cu_seqlens_q
    max_pages_per_seq = int(page_table.shape[1]) if paged_kv else 1
    qsplit_arg, has_qsplit = _qsplit_arg(k2q_q_indices, k2q_qsplit_indices)

    lse_partial = torch.full(
        (int(topk), int(total_q), int(head_q)),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    grid_partial = (int(head_kv) * int(total_rows) * int(num_row_chunks), int(qhead_per_kv))
    _sparse_attn_csr_lse_fp8_kernel[grid_partial](
        q,
        k,
        q2k,
        k2q_row_ptr,
        k2q_q_indices,
        qsplit_arg,
        row_batch,
        row_kv_block,
        page_table_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        lse_partial,
        float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
        int(total_q),
        int(total_rows),
        int(head_q),
        int(head_kv),
        int(qhead_per_kv),
        int(topk),
        int(blk_kv),
        int(max_pages_per_seq),
        int(num_row_chunks),
        bool(has_qsplit),
        bool(causal),
        bool(paged_kv),
        int(dim),
        BLOCK_M=block_m,
        num_warps=8,
    )
    grid_combine = (triton.cdiv(int(total_q), block_m), int(head_q))
    _sparse_attn_lse_combine_kernel[grid_combine](
        lse_partial,
        lse,
        int(total_q),
        int(head_q),
        int(topk),
        BLOCK_M=block_m,
        num_warps=8,
    )
    acc_out = torch.zeros((total_q, head_q, dim), device=q.device, dtype=torch.float32)
    _sparse_attn_csr_accum_fp8_kernel[grid_partial](
        q,
        k,
        v,
        q2k,
        k2q_row_ptr,
        k2q_q_indices,
        qsplit_arg,
        row_batch,
        row_kv_block,
        page_table_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        lse,
        acc_out,
        float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
        int(total_q),
        int(total_rows),
        int(head_q),
        int(head_kv),
        int(qhead_per_kv),
        int(topk),
        int(blk_kv),
        int(max_pages_per_seq),
        int(num_row_chunks),
        bool(has_qsplit),
        bool(causal),
        bool(paged_kv),
        int(dim),
        BLOCK_M=block_m,
        num_warps=8,
    )
    out = torch.empty((total_q, head_q, dim), device=q.device, dtype=torch.bfloat16)
    _sparse_attn_cast_acc_kernel[grid_combine](
        acc_out,
        out,
        int(total_q),
        int(head_q),
        int(dim),
        BLOCK_M=block_m,
        num_warps=8,
    )
    if return_softmax_lse:
        return out, lse
    return out


def _dequant_nvfp4_to_bf16(
    src: torch.Tensor,
    scale_128x4: torch.Tensor,
    global_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """Materialize packed NVFP4 rows as BF16 for the fast CSR attention path."""

    if src.dtype != torch.uint8:
        raise TypeError("NVFP4 source must be packed uint8")
    if src.shape[-1] != 64:
        raise NotImplementedError("SM120 NVFP4 dequant currently supports D=128 packed as 64 bytes")
    src_c = src.contiguous()
    scale_c = scale_128x4.contiguous()
    out_shape = (*src_c.shape[:-1], 128)
    out = torch.empty(out_shape, device=src_c.device, dtype=torch.bfloat16)
    total_rows = int(src_c.numel() // 64)
    global_arg = global_scale if global_scale is not None else out
    fp4_lut, fp8_lut = _nvfp4_luts(src_c.device)
    _nvfp4_dequant_to_bf16_kernel[(total_rows,)](
        src_c,
        scale_c,
        global_arg,
        fp4_lut,
        fp8_lut,
        out,
        int(total_rows),
        bool(global_scale is not None),
        128,
        BLOCK_D=128,
        num_warps=4,
    )
    return out


class _SparseAttentionSm120Autograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q2k: torch.Tensor,
        k2q_row_ptr: torch.Tensor,
        k2q_q_indices: torch.Tensor,
        k2q_qsplit_indices_arg: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table_arg: torch.Tensor,
        softmax_scale: float,
        topk: int,
        blk_kv: int,
        causal: bool,
        paged_kv: bool,
        has_qsplit: bool,
    ) -> torch.Tensor:
        page_table = page_table_arg if paged_kv else None
        qsplit = k2q_qsplit_indices_arg if has_qsplit else None
        total_q, head_q, dim = q.shape
        mode = _forward_mode()
        common = dict(
            topk=int(topk),
            blk_kv=int(blk_kv),
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
            return_softmax_lse=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table,
        )
        if _should_recompute(mode, int(total_q), int(head_q), int(dim), int(topk)):
            out, lse = _sparse_attention_csr_varlen_triton_recompute(
                q, k, v, q2k, k2q_row_ptr, k2q_q_indices, qsplit, **common
            )
        elif mode == "chunked":
            out, lse = _sparse_attention_csr_varlen_triton_two_phase_chunked(
                q, k, v, q2k, k2q_row_ptr, k2q_q_indices, qsplit,
                q_chunk=int(os.environ.get("FMHA_SM120_Q_CHUNK", "4096")),
                **common,
            )
        else:
            out, lse = _sparse_attention_csr_varlen_triton_two_phase(
                q, k, v, q2k, k2q_row_ptr, k2q_q_indices, qsplit, **common
            )
        ctx.save_for_backward(
            q,
            k,
            v,
            q2k,
            k2q_row_ptr,
            k2q_q_indices,
            k2q_qsplit_indices_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            page_table_arg,
            out,
            lse,
        )
        ctx.softmax_scale = float(softmax_scale)
        ctx.topk = int(topk)
        ctx.blk_kv = int(blk_kv)
        ctx.causal = bool(causal)
        ctx.paged_kv = bool(paged_kv)
        ctx.has_qsplit = bool(has_qsplit)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (
            q,
            k,
            v,
            q2k,
            k2q_row_ptr,
            k2q_q_indices,
            k2q_qsplit_indices_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            page_table_arg,
            out,
            lse,
        ) = ctx.saved_tensors
        total_q, head_q, dim = q.shape
        head_kv = k.shape[1]
        max_batch = int(cu_seqlens_q.numel() - 1)
        max_pages_per_seq = int(page_table_arg.shape[1]) if ctx.paged_kv else 1
        backward_mode = os.environ.get("FMHA_SM120_BACKWARD_MODE", "csr").strip().lower()
        if backward_mode not in {"csr", "row"}:
            raise ValueError(f"FMHA_SM120_BACKWARD_MODE must be csr or row, got {backward_mode!r}")
        if backward_mode == "row":
            dq_f = torch.empty(q.shape, device=q.device, dtype=torch.float32)
            dk_f = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
            dv_f = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
            grid = (int(total_q), int(head_q))
            _sparse_attn_bwd_row_bf16_kernel[grid](
                q,
                k,
                v,
                q2k,
                page_table_arg,
                cu_seqlens_q,
                cu_seqlens_k,
                out,
                lse,
                dout.contiguous(),
                dq_f,
                dk_f,
                dv_f,
                float(ctx.softmax_scale),
                int(total_q),
                int(head_q),
                int(head_kv),
                int(head_q // head_kv),
                int(ctx.topk),
                int(ctx.blk_kv),
                int(max_batch),
                int(max_pages_per_seq),
                bool(ctx.causal),
                bool(ctx.paged_kv),
                int(dim),
                num_warps=8,
            )
        else:
            dq_f = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
            dk_f = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
            dv_f = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
            block_m = _csr_block_m(64)
            total_rows, row_batch, row_kv_block, num_row_chunks = _csr_launch_params(
                k2q_row_ptr, cu_seqlens_k, int(ctx.blk_kv), block_m
            )
            grid = (int(head_kv) * int(total_rows) * int(num_row_chunks), int(head_q // head_kv))
            _sparse_attn_bwd_csr_bf16_kernel[grid](
                q,
                k,
                v,
                k2q_row_ptr,
                k2q_q_indices,
                k2q_qsplit_indices_arg,
                row_batch,
                row_kv_block,
                page_table_arg,
                cu_seqlens_q,
                cu_seqlens_k,
                out,
                lse,
                dout.contiguous(),
                dq_f,
                dk_f,
                dv_f,
                float(ctx.softmax_scale),
                int(total_q),
                int(total_rows),
                int(head_q),
                int(head_kv),
                int(head_q // head_kv),
                int(ctx.topk),
                int(ctx.blk_kv),
                int(max_pages_per_seq),
                int(num_row_chunks),
                bool(ctx.has_qsplit),
                bool(ctx.causal),
                bool(ctx.paged_kv),
                int(dim),
                BLOCK_M=block_m,
                num_warps=8,
            )
        return (
            dq_f.to(q.dtype),
            dk_f.to(k.dtype),
            dv_f.to(v.dtype),
        ) + (None,) * 13


class _SparseAttentionNvfp4KvSm120Autograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k_ref: torch.Tensor,
        v_ref: torch.Tensor,
        k_packed: torch.Tensor,
        v_packed: torch.Tensor,
        k_scale_128x4: torch.Tensor,
        v_scale_128x4: torch.Tensor,
        k_global_scale_arg: torch.Tensor,
        v_global_scale_arg: torch.Tensor,
        q2k_indices: torch.Tensor,
        k2q_row_ptr: torch.Tensor,
        k2q_q_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table_arg: torch.Tensor,
        softmax_scale: float,
        topk: int,
        blk_kv: int,
        causal: bool,
        paged_kv: bool,
        has_k_global_scale: bool,
        has_v_global_scale: bool,
    ):
        out, lse = sparse_attention_nvfp4_kv_triton(
            q,
            k_packed,
            v_packed,
            k_scale_128x4,
            v_scale_128x4,
            k_global_scale_arg if has_k_global_scale else None,
            v_global_scale_arg if has_v_global_scale else None,
            k2q_row_ptr,
            k2q_q_indices,
            q2k_indices,
            topk=int(topk),
            blk_kv=int(blk_kv),
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
            return_softmax_lse=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table_arg if paged_kv else None,
            seqused_k=None,
        )
        ctx.save_for_backward(
            q,
            k_packed,
            v_packed,
            k_scale_128x4,
            v_scale_128x4,
            k_global_scale_arg,
            v_global_scale_arg,
            q2k_indices,
            k2q_row_ptr,
            k2q_q_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            page_table_arg,
            out,
            lse,
        )
        ctx.k_ref_shape = tuple(k_ref.shape)
        ctx.v_ref_shape = tuple(v_ref.shape)
        ctx.k_ref_dtype = k_ref.dtype
        ctx.v_ref_dtype = v_ref.dtype
        ctx.softmax_scale = float(softmax_scale)
        ctx.topk = int(topk)
        ctx.blk_kv = int(blk_kv)
        ctx.causal = bool(causal)
        ctx.paged_kv = bool(paged_kv)
        ctx.has_k_global_scale = bool(has_k_global_scale)
        ctx.has_v_global_scale = bool(has_v_global_scale)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (
            q,
            k_packed,
            v_packed,
            k_scale_128x4,
            v_scale_128x4,
            k_global_scale_arg,
            v_global_scale_arg,
            q2k_indices,
            k2q_row_ptr,
            k2q_q_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            page_table_arg,
            out,
            lse,
        ) = ctx.saved_tensors
        total_q, head_q, dim = q.shape
        head_kv = int(ctx.k_ref_shape[1])
        dq_f = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dk_f = torch.zeros(ctx.k_ref_shape, device=q.device, dtype=torch.float32)
        dv_f = torch.zeros(ctx.v_ref_shape, device=q.device, dtype=torch.float32)
        block_m = _csr_block_m(64)
        total_rows, row_batch, row_kv_block, num_row_chunks = _csr_launch_params(
            k2q_row_ptr, cu_seqlens_k, int(ctx.blk_kv), block_m
        )
        max_pages_per_seq = int(page_table_arg.shape[1]) if ctx.paged_kv else 1
        grid = (int(head_kv) * int(total_rows) * int(num_row_chunks), int(head_q // head_kv))
        _sparse_attn_bwd_csr_nvfp4_kernel[grid](
            q,
            k_packed,
            v_packed,
            k_scale_128x4,
            v_scale_128x4,
            k_global_scale_arg,
            v_global_scale_arg,
            q2k_indices,
            k2q_row_ptr,
            k2q_q_indices,
            row_batch,
            row_kv_block,
            page_table_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            out,
            lse,
            dout.contiguous(),
            dq_f,
            dk_f,
            dv_f,
            float(ctx.softmax_scale),
            int(total_q),
            int(total_rows),
            int(head_q),
            int(head_kv),
            int(head_q // head_kv),
            int(ctx.topk),
            int(ctx.blk_kv),
            int(max_pages_per_seq),
            int(num_row_chunks),
            bool(ctx.causal),
            bool(ctx.paged_kv),
            bool(ctx.has_k_global_scale),
            bool(ctx.has_v_global_scale),
            int(dim),
            BLOCK_M=block_m,
            num_warps=8,
        )
        return (
            dq_f.to(q.dtype),
            dk_f.to(ctx.k_ref_dtype),
            dv_f.to(ctx.v_ref_dtype),
        ) + (None,) * 19


class _SparseAttentionFp8KvSm120Autograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k_ref: torch.Tensor,
        v_ref: torch.Tensor,
        k_fp8_u8: torch.Tensor,
        v_fp8_u8: torch.Tensor,
        q2k_indices: torch.Tensor,
        k2q_row_ptr: torch.Tensor,
        k2q_q_indices: torch.Tensor,
        k2q_qsplit_indices_arg: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        page_table_arg: torch.Tensor,
        softmax_scale: float,
        topk: int,
        blk_kv: int,
        causal: bool,
        paged_kv: bool,
        has_qsplit: bool,
    ):
        out, lse = _sparse_attention_csr_varlen_triton_recompute_fp8(
            q,
            k_fp8_u8,
            v_fp8_u8,
            q2k_indices,
            k2q_row_ptr,
            k2q_q_indices,
            k2q_qsplit_indices_arg if has_qsplit else None,
            topk=int(topk),
            blk_kv=int(blk_kv),
            causal=bool(causal),
            softmax_scale=float(softmax_scale),
            return_softmax_lse=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table_arg if paged_kv else None,
        )
        ctx.save_for_backward(
            q,
            k_fp8_u8,
            v_fp8_u8,
            q2k_indices,
            k2q_row_ptr,
            k2q_q_indices,
            k2q_qsplit_indices_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            page_table_arg,
            out,
            lse,
        )
        ctx.k_ref_shape = tuple(k_ref.shape)
        ctx.v_ref_shape = tuple(v_ref.shape)
        ctx.k_ref_dtype = k_ref.dtype
        ctx.v_ref_dtype = v_ref.dtype
        ctx.softmax_scale = float(softmax_scale)
        ctx.topk = int(topk)
        ctx.blk_kv = int(blk_kv)
        ctx.causal = bool(causal)
        ctx.paged_kv = bool(paged_kv)
        ctx.has_qsplit = bool(has_qsplit)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        (
            q,
            k_fp8_u8,
            v_fp8_u8,
            q2k,
            k2q_row_ptr,
            k2q_q_indices,
            k2q_qsplit_indices_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            page_table_arg,
            out,
            lse,
        ) = ctx.saved_tensors
        total_q, head_q, dim = q.shape
        head_kv = int(ctx.k_ref_shape[1])
        dq_f = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        dk_f = torch.zeros(ctx.k_ref_shape, device=q.device, dtype=torch.float32)
        dv_f = torch.zeros(ctx.v_ref_shape, device=q.device, dtype=torch.float32)
        block_m = _csr_block_m(64)
        total_rows, row_batch, row_kv_block, num_row_chunks = _csr_launch_params(
            k2q_row_ptr, cu_seqlens_k, int(ctx.blk_kv), block_m
        )
        max_pages_per_seq = int(page_table_arg.shape[1]) if ctx.paged_kv else 1
        grid = (int(head_kv) * int(total_rows) * int(num_row_chunks), int(head_q // head_kv))
        _sparse_attn_bwd_csr_fp8_kernel[grid](
            q,
            k_fp8_u8,
            v_fp8_u8,
            k2q_row_ptr,
            k2q_q_indices,
            k2q_qsplit_indices_arg,
            row_batch,
            row_kv_block,
            page_table_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            out,
            lse,
            dout.contiguous(),
            dq_f,
            dk_f,
            dv_f,
            float(ctx.softmax_scale),
            int(total_q),
            int(total_rows),
            int(head_q),
            int(head_kv),
            int(head_q // head_kv),
            int(ctx.topk),
            int(ctx.blk_kv),
            int(max_pages_per_seq),
            int(num_row_chunks),
            bool(ctx.has_qsplit),
            bool(ctx.causal),
            bool(ctx.paged_kv),
            int(dim),
            BLOCK_M=block_m,
            num_warps=8,
        )
        return (
            dq_f.to(q.dtype),
            dk_f.to(ctx.k_ref_dtype),
            dv_f.to(ctx.v_ref_dtype),
        ) + (None,) * 15


def sparse_attention_csr_varlen_triton_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    *,
    q2k_indices: Optional[torch.Tensor],
    k2q_qsplit_indices: Optional[torch.Tensor] = None,
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
) -> torch.Tensor:
    if q2k_indices is None:
        q2k_indices = _reconstruct_q2k_from_k2q_csr(
            k2q_row_ptr,
            k2q_q_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            topk=int(topk),
            blk_kv=int(blk_kv),
            total_q=int(q.shape[0]),
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise NotImplementedError("SM120 Triton autograd backend currently supports BF16 Q/K/V only")
    if q.shape[-1] != 128:
        raise NotImplementedError("SM120 Triton autograd backend currently supports D=128 only")
    if _forward_mode() == "qstat":
        # Q-stationary single-pass kernels: fastest when neighboring queries
        # select overlapping blocks (the realistic MSA regime); no atomics, so
        # gradients are deterministic. Constraints are enforced loudly.
        if not causal:
            raise NotImplementedError("qstat mode supports causal=True only")
        if page_table is not None:
            raise NotImplementedError("qstat mode does not support paged KV")
        from src.sm120.qstat import sparse_attention_qstat

        return sparse_attention_qstat(
            q,
            k,
            v,
            q2k_indices,
            k2q_row_ptr,
            k2q_q_indices,
            topk=int(topk),
            blk_kv=int(blk_kv),
            softmax_scale=softmax_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )
    if page_table is not None:
        page_table_arg = page_table.contiguous()
        paged_kv = True
    else:
        page_table_arg = cu_seqlens_q
        paged_kv = False
    return _SparseAttentionSm120Autograd.apply(
        q,
        k,
        v,
        q2k_indices.contiguous(),
        k2q_row_ptr,
        k2q_q_indices,
        k2q_qsplit_indices.contiguous() if k2q_qsplit_indices is not None else k2q_q_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table_arg,
        float(softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)),
        int(topk),
        int(blk_kv),
        bool(causal),
        bool(paged_kv),
        bool(k2q_qsplit_indices is not None),
    )


def sparse_attention_nvfp4_kv_triton_autograd(
    q: torch.Tensor,
    k_ref: torch.Tensor,
    v_ref: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    k_scale_128x4: torch.Tensor,
    v_scale_128x4: torch.Tensor,
    k_global_scale: Optional[torch.Tensor],
    v_global_scale: Optional[torch.Tensor],
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    *,
    q2k_indices: Optional[torch.Tensor],
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
) -> torch.Tensor:
    if q2k_indices is None:
        q2k_indices = _reconstruct_q2k_from_k2q_csr(
            k2q_row_ptr,
            k2q_q_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            topk=int(topk),
            blk_kv=int(blk_kv),
            total_q=int(q.shape[0]),
        )
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("SM120 NVFP4 training backend currently supports BF16 Q only")
    if k_ref.dtype not in (torch.bfloat16, torch.float16) or v_ref.dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError("SM120 NVFP4 training backend expects BF16/FP16 logical K/V references")
    if k_packed.dtype != torch.uint8 or v_packed.dtype != torch.uint8:
        raise TypeError("SM120 NVFP4 training backend expects packed uint8 K/V")
    if q.shape[-1] != 128:
        raise NotImplementedError("SM120 NVFP4 training backend currently supports D=128 only")
    if page_table is not None:
        page_table_arg = page_table.contiguous()
        paged_kv = True
    else:
        page_table_arg = cu_seqlens_q
        paged_kv = False
    k_global_arg = k_global_scale if k_global_scale is not None else q
    v_global_arg = v_global_scale if v_global_scale is not None else q
    return _SparseAttentionNvfp4KvSm120Autograd.apply(
        q.contiguous(),
        k_ref,
        v_ref,
        k_packed.contiguous(),
        v_packed.contiguous(),
        k_scale_128x4.contiguous(),
        v_scale_128x4.contiguous(),
        k_global_arg.contiguous(),
        v_global_arg.contiguous(),
        q2k_indices.contiguous(),
        k2q_row_ptr.contiguous(),
        k2q_q_indices.contiguous(),
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        page_table_arg.contiguous(),
        float(softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)),
        int(topk),
        int(blk_kv),
        bool(causal),
        bool(paged_kv),
        bool(k_global_scale is not None),
        bool(v_global_scale is not None),
    )


def sparse_attention_fp8_kv_triton_autograd(
    q: torch.Tensor,
    k_ref: torch.Tensor,
    v_ref: torch.Tensor,
    k_fp8_u8: torch.Tensor,
    v_fp8_u8: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    *,
    q2k_indices: Optional[torch.Tensor],
    k2q_qsplit_indices: Optional[torch.Tensor] = None,
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
) -> torch.Tensor:
    if q2k_indices is None:
        q2k_indices = _reconstruct_q2k_from_k2q_csr(
            k2q_row_ptr,
            k2q_q_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            topk=int(topk),
            blk_kv=int(blk_kv),
            total_q=int(q.shape[0]),
        )
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("SM120 FP8 K/V training backend currently supports BF16 Q only")
    if k_ref.dtype not in (torch.bfloat16, torch.float16) or v_ref.dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError("SM120 FP8 K/V training backend expects BF16/FP16 logical K/V references")
    if k_fp8_u8.dtype != torch.uint8 or v_fp8_u8.dtype != torch.uint8:
        raise TypeError("SM120 FP8 K/V training backend expects uint8 views of torch.float8_e4m3fn K/V")
    if q.shape[-1] != 128:
        raise NotImplementedError("SM120 FP8 K/V training backend currently supports D=128 only")
    if page_table is not None:
        page_table_arg = page_table.contiguous()
        paged_kv = True
    else:
        page_table_arg = cu_seqlens_q
        paged_kv = False
    return _SparseAttentionFp8KvSm120Autograd.apply(
        q.contiguous(),
        k_ref,
        v_ref,
        k_fp8_u8.contiguous(),
        v_fp8_u8.contiguous(),
        q2k_indices.contiguous(),
        k2q_row_ptr.contiguous(),
        k2q_q_indices.contiguous(),
        k2q_qsplit_indices.contiguous() if k2q_qsplit_indices is not None else k2q_q_indices,
        cu_seqlens_q.contiguous(),
        cu_seqlens_k.contiguous(),
        page_table_arg,
        float(softmax_scale if softmax_scale is not None else (q.shape[-1] ** -0.5)),
        int(topk),
        int(blk_kv),
        bool(causal),
        bool(paged_kv),
        bool(k2q_qsplit_indices is not None),
    )


def sparse_attention_csr_varlen_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    *,
    q2k_indices: Optional[torch.Tensor] = None,
    k2q_qsplit_indices: Optional[torch.Tensor] = None,
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    return_softmax_lse: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if seqused_k is not None:
        raise NotImplementedError("SM120 Triton backend currently requires cu_seqlens_k effective lengths")
    fp8_kv_cache = (
        q.dtype == torch.bfloat16
        and k.dtype == torch.float8_e4m3fn
        and v.dtype == torch.float8_e4m3fn
    )
    if not (
        (q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16)
        or fp8_kv_cache
    ):
        raise NotImplementedError(
            "SM120 Triton backend currently supports BF16 Q/K/V or BF16 Q with FP8 E4M3 K/V"
        )
    total_q, head_q, dim = q.shape
    head_kv = k.shape[1]
    if dim != 128:
        raise NotImplementedError(f"SM120 Triton backend currently supports D=128, got {dim}")
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by k.shape[1]")
    q2k = (
        q2k_indices.contiguous()
        if q2k_indices is not None
        else _reconstruct_q2k_from_k2q_csr(
            k2q_row_ptr,
            k2q_q_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            topk=int(topk),
            blk_kv=int(blk_kv),
            total_q=int(total_q),
        )
    )
    # Fail fast on inconsistent metadata before any kernel launch.
    if int(cu_seqlens_q[-1].item()) - int(cu_seqlens_q[0].item()) != int(total_q):
        raise ValueError("cu_seqlens_q does not match q.shape[0]")
    mode = _forward_mode()
    if mode == "qstat":
        if not causal:
            raise NotImplementedError("qstat mode supports causal=True only")
        if page_table is not None:
            raise NotImplementedError("qstat mode does not support paged KV")
        if fp8_kv_cache:
            raise NotImplementedError(
                "qstat mode takes BF16 K/V here; call sparse_attention_qstat_fp8 directly"
            )
        from src.sm120.qstat import sparse_attention_qstat

        return sparse_attention_qstat(
            q,
            k,
            v,
            q2k,
            k2q_row_ptr,
            k2q_q_indices,
            topk=int(topk),
            blk_kv=int(blk_kv),
            softmax_scale=softmax_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            return_softmax_lse=return_softmax_lse,
        )
    common = dict(
        topk=topk,
        blk_kv=blk_kv,
        causal=causal,
        softmax_scale=softmax_scale,
        return_softmax_lse=return_softmax_lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        page_table=page_table,
    )
    if _should_recompute(mode, int(total_q), int(head_q), int(dim), int(topk)):
        return _sparse_attention_csr_varlen_triton_recompute(
            q, k, v, q2k, k2q_row_ptr, k2q_q_indices, k2q_qsplit_indices, **common
        )
    if mode == "chunked":
        return _sparse_attention_csr_varlen_triton_two_phase_chunked(
            q, k, v, q2k, k2q_row_ptr, k2q_q_indices, k2q_qsplit_indices,
            q_chunk=int(os.environ.get("FMHA_SM120_Q_CHUNK", "4096")),
            **common,
        )
    if mode != "row":
        return _sparse_attention_csr_varlen_triton_two_phase(
            q, k, v, q2k, k2q_row_ptr, k2q_q_indices, k2q_qsplit_indices, **common
        )
    # row: one program per (query, head); a straightforward kernel kept for
    # debugging the tiled paths.
    out = torch.empty((total_q, head_q, dim), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    max_batch = int(cu_seqlens_q.numel() - 1)
    paged_kv = page_table is not None
    max_pages_per_seq = int(page_table.shape[1]) if paged_kv else 1
    page_table_arg = page_table if paged_kv else cu_seqlens_q
    grid = (int(total_q), int(head_q))
    _sparse_attn_dense_bf16_kernel[grid](
        q,
        k,
        v,
        q2k,
        page_table_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        lse,
        float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
        int(total_q),
        int(head_q),
        int(head_kv),
        int(head_q // head_kv),
        int(topk),
        int(blk_kv),
        max_batch,
        max_pages_per_seq,
        bool(causal),
        bool(paged_kv),
        int(dim),
        num_warps=8,
    )
    if return_softmax_lse:
        return out, lse
    return out


def sparse_attention_nvfp4_kv_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale_128x4: torch.Tensor,
    v_scale_128x4: torch.Tensor,
    k_global_scale: Optional[torch.Tensor],
    v_global_scale: Optional[torch.Tensor],
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    q2k_indices: Optional[torch.Tensor] = None,
    *,
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    return_softmax_lse: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if seqused_k is not None:
        raise NotImplementedError("SM120 NVFP4 backend currently requires cu_seqlens_k effective lengths")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("SM120 NVFP4 backend currently supports BF16 Q only")
    if k.dtype != torch.uint8 or v.dtype != torch.uint8:
        raise TypeError("SM120 NVFP4 backend expects packed uint8 K/V")
    total_q, head_q, dim = q.shape
    if dim != 128:
        raise NotImplementedError(f"SM120 NVFP4 backend currently supports D=128, got {dim}")
    head_kv = int(k.shape[1])
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by K/V head count")
    q2k = (
        q2k_indices.contiguous()
        if q2k_indices is not None
        else _reconstruct_q2k_from_k2q_csr(
            k2q_row_ptr,
            k2q_q_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            topk=int(topk),
            blk_kv=int(blk_kv),
            total_q=int(total_q),
        )
    )
    mode = os.environ.get("FMHA_SM120_NVFP4_MODE", "csr").strip().lower()
    if mode not in {"csr", "csr_scalar", "row"}:
        raise ValueError(f"FMHA_SM120_NVFP4_MODE must be one of csr/csr_scalar/row, got {mode!r}")
    if mode == "csr":
        k_bf16 = _dequant_nvfp4_to_bf16(k, k_scale_128x4, k_global_scale)
        v_bf16 = _dequant_nvfp4_to_bf16(v, v_scale_128x4, v_global_scale)
        return _sparse_attention_csr_varlen_triton_recompute(
            q,
            k_bf16,
            v_bf16,
            q2k,
            k2q_row_ptr,
            k2q_q_indices,
            None,
            topk=topk,
            blk_kv=blk_kv,
            causal=causal,
            softmax_scale=softmax_scale,
            return_softmax_lse=return_softmax_lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table,
        )
    out = torch.empty((total_q, head_q, dim), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    paged_kv = page_table is not None
    page_table_arg = page_table if paged_kv else cu_seqlens_q
    max_pages_per_seq = int(page_table.shape[1]) if paged_kv else 1
    max_batch = int(cu_seqlens_q.numel() - 1)
    k_global_arg = k_global_scale if k_global_scale is not None else lse
    v_global_arg = v_global_scale if v_global_scale is not None else lse
    if mode == "csr_scalar":
        block_m = _csr_block_m(32)
        total_rows, row_batch, row_kv_block, num_row_chunks = _csr_launch_params(
            k2q_row_ptr, cu_seqlens_k, int(blk_kv), block_m
        )
        lse_partial = torch.full(
            (int(topk), int(total_q), int(head_q)),
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        grid_partial = (int(head_kv) * int(total_rows) * int(num_row_chunks), int(head_q // head_kv))
        _sparse_attn_csr_lse_nvfp4_kernel[grid_partial](
            q,
            k,
            k_scale_128x4,
            k_global_arg,
            q2k,
            k2q_row_ptr,
            k2q_q_indices,
            row_batch,
            row_kv_block,
            page_table_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            lse_partial,
            float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
            int(total_q),
            int(total_rows),
            int(head_q),
            int(head_kv),
            int(head_q // head_kv),
            int(topk),
            int(blk_kv),
            int(max_pages_per_seq),
            int(num_row_chunks),
            bool(causal),
            bool(paged_kv),
            bool(k_global_scale is not None),
            int(dim),
            BLOCK_M=block_m,
            num_warps=8,
        )
        grid_combine = (triton.cdiv(int(total_q), block_m), int(head_q))
        _sparse_attn_lse_combine_kernel[grid_combine](
            lse_partial,
            lse,
            int(total_q),
            int(head_q),
            int(topk),
            BLOCK_M=block_m,
            num_warps=8,
        )
        acc_out = torch.zeros((total_q, head_q, dim), device=q.device, dtype=torch.float32)
        _sparse_attn_csr_accum_nvfp4_kernel[grid_partial](
            q,
            k,
            v,
            k_scale_128x4,
            v_scale_128x4,
            k_global_arg,
            v_global_arg,
            q2k,
            k2q_row_ptr,
            k2q_q_indices,
            row_batch,
            row_kv_block,
            page_table_arg,
            cu_seqlens_q,
            cu_seqlens_k,
            lse,
            acc_out,
            float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
            int(total_q),
            int(total_rows),
            int(head_q),
            int(head_kv),
            int(head_q // head_kv),
            int(topk),
            int(blk_kv),
            int(max_pages_per_seq),
            int(num_row_chunks),
            bool(causal),
            bool(paged_kv),
            bool(k_global_scale is not None),
            bool(v_global_scale is not None),
            int(dim),
            BLOCK_M=block_m,
            num_warps=8,
        )
        _sparse_attn_cast_acc_kernel[grid_combine](
            acc_out,
            out,
            int(total_q),
            int(head_q),
            int(dim),
            BLOCK_M=block_m,
            num_warps=8,
        )
        if return_softmax_lse:
            return out, lse
        return out
    grid = (int(total_q), int(head_q))
    _sparse_attn_dense_nvfp4_kernel[grid](
        q,
        k,
        v,
        k_scale_128x4,
        v_scale_128x4,
        k_global_arg,
        v_global_arg,
        q2k,
        page_table_arg,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        lse,
        float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
        int(total_q),
        int(head_q),
        int(head_kv),
        int(head_q // head_kv),
        int(topk),
        int(blk_kv),
        int(max_batch),
        int(max_pages_per_seq),
        bool(causal),
        bool(paged_kv),
        bool(k_global_scale is not None),
        bool(v_global_scale is not None),
        int(dim),
        num_warps=8,
    )
    if return_softmax_lse:
        return out, lse
    return out


@triton.jit
def _sparse_decode_paged_fp8_kernel(
    q,
    k,
    v,
    q2k,
    page_table,
    seqused_k,
    out,
    lse_out,
    softmax_scale: tl.constexpr,
    total_q: tl.constexpr,
    head_q: tl.constexpr,
    head_kv: tl.constexpr,
    qhead_per_kv: tl.constexpr,
    seqlen_q: tl.constexpr,
    blk_kv: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    topk: tl.constexpr,
    sparse: tl.constexpr,
    quantize_p_fp8: tl.constexpr,
    dim: tl.constexpr,
):
    q_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    batch_idx = q_idx // seqlen_q
    q_local = q_idx - batch_idx * seqlen_q
    kv_head = q_head // qhead_per_kv

    offs_d = tl.arange(0, 128)
    offs_n = tl.arange(0, 128)
    d_mask = offs_d < dim
    q_vec = tl.load(q + (q_idx * head_q + q_head) * dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    used_k = tl.load(seqused_k + batch_idx)
    causal_limit = q_local + (used_k - seqlen_q)

    m_i = tl.full((), -float("inf"), tl.float32)
    l_i = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((128,), tl.float32)

    for slot in tl.static_range(0, topk):
        logical_page = slot
        selected_valid = True
        if sparse:
            selected = tl.load(
                q2k + (kv_head * total_q + q_idx) * topk + slot,
                mask=q_idx < total_q,
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
        k_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
        k_tile = tl.load(k_ptrs, mask=token_valid[None, :] & d_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k_tile * q_vec[:, None], axis=0) * softmax_scale
        scores = tl.where(token_valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(token_valid, p, 0.0)
        if quantize_p_fp8:
            # Match the SM100 decode contract: QK/LSE are fp32, but the
            # unnormalized probabilities are rounded to e4m3 before PV.
            p_pv = p.to(tl.float8e4nv).to(tl.float32)
        else:
            p_pv = p
        v_ptrs = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        v_tile = tl.load(v_ptrs, mask=token_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p_pv[:, None] * v_tile, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    has_value = l_i > 0.0
    safe_l = tl.where(has_value, l_i, 1.0)
    out_vec = tl.where(has_value, acc / safe_l, 0.0)
    tl.store(out + (q_idx * head_q + q_head) * dim + offs_d, out_vec, mask=d_mask)
    tl.store(
        lse_out + q_idx * head_q + q_head,
        tl.where(has_value, tl.log(l_i) + m_i, -30000.0),  # finite empty-row sentinel
    )


@triton.jit
def _sparse_decode_paged_fp8_split_kernel(
    q,
    k,
    v,
    q2k,
    page_table,
    seqused_k,
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
    q_vec = tl.load(q + (q_idx * head_q + q_head) * dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    used_k = tl.load(seqused_k + batch_idx)
    causal_limit = q_local + (used_k - seqlen_q)

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
        k_ptrs = k + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[None, :]) * dim + offs_d[:, None])
        k_tile = tl.load(k_ptrs, mask=token_valid[None, :] & d_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k_tile * q_vec[:, None], axis=0) * softmax_scale
        scores = tl.where(token_valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(token_valid, p, 0.0)
        if quantize_p_fp8:
            p_pv = p.to(tl.float8e4nv).to(tl.float32)
        else:
            p_pv = p
        v_ptrs = v + (((physical_page * head_kv + kv_head) * blk_kv + offs_n[:, None]) * dim + offs_d[None, :])
        v_tile = tl.load(v_ptrs, mask=token_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p_pv[:, None] * v_tile, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    has_value = l_i > 0.0
    safe_l = tl.where(has_value, l_i, 1.0)
    out_vec = tl.where(has_value, acc / safe_l, 0.0)
    tl.store(
        o_partial + ((split_idx * total_q + q_idx) * head_q + q_head) * dim + offs_d,
        out_vec,
        mask=d_mask,
    )
    tl.store(
        lse_partial + (split_idx * total_q + q_idx) * head_q + q_head,
        tl.where(has_value, tl.log(l_i) + m_i, -30000.0),  # finite empty-row sentinel
    )


@triton.jit
def _sparse_decode_split_combine_kernel(
    o_partial,
    lse_partial,
    out,
    lse_out,
    total_q: tl.constexpr,
    head_q: tl.constexpr,
    num_splits: tl.constexpr,
    dim: tl.constexpr,
):
    q_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    offs_d = tl.arange(0, 128)
    d_mask = offs_d < dim

    m = tl.full((), -float("inf"), tl.float32)
    for split_idx in tl.static_range(0, num_splits):
        lse = tl.load(lse_partial + (split_idx * total_q + q_idx) * head_q + q_head)
        m = tl.maximum(m, lse)

    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((128,), tl.float32)
    for split_idx in tl.static_range(0, num_splits):
        lse = tl.load(lse_partial + (split_idx * total_q + q_idx) * head_q + q_head)
        w = tl.exp(lse - m)
        w = tl.where(lse > -float("inf"), w, 0.0)
        part = tl.load(
            o_partial + ((split_idx * total_q + q_idx) * head_q + q_head) * dim + offs_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        acc += w * part
        denom += w

    has_value = denom > 0.0
    out_vec = tl.where(has_value, acc / denom, 0.0)
    tl.store(out + (q_idx * head_q + q_head) * dim + offs_d, out_vec, mask=d_mask)
    tl.store(
        lse_out + q_idx * head_q + q_head,
        tl.where(has_value, tl.log(denom) + m, -30000.0),  # finite empty-row sentinel
    )


def sparse_decode_paged_fp8_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: Optional[torch.Tensor],
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    return_softmax_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if not causal:
        raise NotImplementedError("SM120 decode backend currently supports causal=True only")
    if q.dtype != torch.float8_e4m3fn or k.dtype != q.dtype or v.dtype != q.dtype:
        raise NotImplementedError("SM120 decode backend currently supports FP8 E4M3 Q/K/V only")
    if q.shape[-1] != 128:
        raise NotImplementedError("SM120 decode backend currently supports D=128 only")
    total_q, head_q, dim = q.shape
    head_kv = int(k.shape[1])
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by k.shape[1]")
    max_pages_per_seq = int(page_table.shape[1])
    if q2k_indices is not None:
        q2k = q2k_indices.contiguous()
        topk = int(q2k.shape[-1])
        sparse = True
    else:
        q2k = page_table
        topk = max_pages_per_seq
        sparse = False
    out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    quantize_p_fp8 = os.environ.get("FMHA_SM120_DECODE_FP8_P", "1").strip().lower() not in {
        "0",
        "false",
        "off",
    }
    split_pages = int(os.environ.get("FMHA_SM120_DECODE_SPLIT_PAGES", "0"))
    if split_pages <= 0 and topk > 32:
        # The page loop is a compile-time unroll (tl.static_range). Dense
        # decode over a long context would unroll to max_pages_per_seq
        # iterations, which explodes Triton compile time, so bound the
        # per-program unroll and let the split-combine kernel do the rest.
        split_pages = 32
    if split_pages > 0 and topk > split_pages:
        num_splits = triton.cdiv(int(topk), int(split_pages))
        o_partial = torch.empty(
            (int(num_splits), int(total_q), int(head_q), int(dim)),
            device=q.device,
            dtype=torch.float32,
        )
        lse_partial = torch.empty(
            (int(num_splits), int(total_q), int(head_q)),
            device=q.device,
            dtype=torch.float32,
        )
        grid_split = (int(total_q), int(head_q), int(num_splits))
        _sparse_decode_paged_fp8_split_kernel[grid_split](
            q,
            k,
            v,
            q2k,
            page_table,
            seqused_k,
            o_partial,
            lse_partial,
            float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
            int(total_q),
            int(head_q),
            int(head_kv),
            int(head_q // head_kv),
            int(seqlen_q),
            int(blk_kv),
            int(max_pages_per_seq),
            int(topk),
            int(split_pages),
            bool(sparse),
            bool(quantize_p_fp8),
            int(dim),
            num_warps=8,
        )
        grid_combine = (int(total_q), int(head_q))
        _sparse_decode_split_combine_kernel[grid_combine](
            o_partial,
            lse_partial,
            out,
            lse,
            int(total_q),
            int(head_q),
            int(num_splits),
            int(dim),
            num_warps=8,
        )
        if return_softmax_lse:
            return out, lse
        return out
    grid = (int(total_q), int(head_q))
    _sparse_decode_paged_fp8_kernel[grid](
        q,
        k,
        v,
        q2k,
        page_table,
        seqused_k,
        out,
        lse,
        float(softmax_scale if softmax_scale is not None else (dim ** -0.5)),
        int(total_q),
        int(head_q),
        int(head_kv),
        int(head_q // head_kv),
        int(seqlen_q),
        int(blk_kv),
        int(max_pages_per_seq),
        int(topk),
        bool(sparse),
        bool(quantize_p_fp8),
        int(dim),
        num_warps=8,
    )
    if return_softmax_lse:
        return out, lse
    return out
