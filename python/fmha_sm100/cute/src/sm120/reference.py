# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""SM120-compatible sparse attention reference backend.

This backend preserves the public CSR varlen sparse-attention contract without
using the SM100-only CuTe/tcgen05/TMEM path.  It is deliberately simple and
serves as a correctness oracle / functional fallback for RTX PRO 6000
Blackwell (sm_120).  A fused SM120 kernel should replace this for production
throughput.
"""

from __future__ import annotations

from typing import Optional

import torch


def _to_int_list(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.detach().cpu().tolist()]


def _row_map(cu_seqlens_k: torch.Tensor, blk_kv: int) -> list[tuple[int, int]]:
    k_cu = _to_int_list(cu_seqlens_k)
    rows_per_batch = [
        (max(k_cu[i + 1] - k_cu[i], 0) + blk_kv - 1) // blk_kv
        for i in range(len(k_cu) - 1)
    ]
    max_rows = max(rows_per_batch, default=0)
    rows: list[tuple[int, int]] = []
    for kv_block in range(max_rows):
        for batch_idx, row_count in enumerate(rows_per_batch):
            if kv_block < row_count:
                rows.append((batch_idx, kv_block))
    return rows


def _reconstruct_q2k_from_k2q_csr(
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    topk: int,
    blk_kv: int,
    total_q: int,
) -> torch.Tensor:
    """Reconstruct [Hkv, total_q, topK] q2k indices from public CSR metadata."""

    head_kv = int(k2q_row_ptr.shape[0])
    q2k = torch.full(
        (head_kv, total_q, topk),
        -1,
        dtype=torch.int32,
        device=k2q_row_ptr.device,
    )
    counts = torch.zeros((head_kv, total_q), dtype=torch.int64, device=k2q_row_ptr.device)
    rows = _row_map(cu_seqlens_k, blk_kv)
    q_cu = _to_int_list(cu_seqlens_q)
    row_ptr_cpu = k2q_row_ptr.detach().cpu()
    q_idx_cpu = k2q_q_indices.detach().cpu()

    for h in range(head_kv):
        for row, (batch_idx, kv_block) in enumerate(rows):
            start = int(row_ptr_cpu[h, row].item())
            end = int(row_ptr_cpu[h, row + 1].item())
            q_base = q_cu[batch_idx]
            for offset in range(start, end):
                q_local = int(q_idx_cpu[h, offset].item())
                if q_local < 0:
                    continue
                q_global = q_base + q_local
                if q_global < 0 or q_global >= total_q:
                    continue
                slot = int(counts[h, q_global].item())
                if slot < topk:
                    q2k[h, q_global, slot] = int(kv_block)
                    counts[h, q_global] += 1
    return q2k


def _query_batch_lookup(cu_seqlens_q: torch.Tensor) -> list[tuple[int, int]]:
    q_cu = _to_int_list(cu_seqlens_q)
    out: list[tuple[int, int]] = []
    for batch_idx in range(len(q_cu) - 1):
        for q_local in range(q_cu[batch_idx + 1] - q_cu[batch_idx]):
            out.append((batch_idx, q_local))
    return out


def _kv_slice_dense(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    batch_idx: int,
    kv_head: int,
    kv_block: int,
    blk_kv: int,
    cu_seqlens_k: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_start = cu_seqlens_k[batch_idx]
    k_end = cu_seqlens_k[batch_idx + 1]
    local_start = kv_block * blk_kv
    local_end = min(local_start + blk_kv, k_end - k_start)
    if local_start >= local_end:
        empty = torch.empty(0, k.shape[-1], device=k.device, dtype=k.dtype)
        pos = torch.empty(0, device=k.device, dtype=torch.int32)
        return empty, empty.to(v.dtype), pos
    global_start = k_start + local_start
    global_end = k_start + local_end
    pos = torch.arange(local_start, local_end, device=k.device, dtype=torch.int32)
    return k[global_start:global_end, kv_head, :], v[global_start:global_end, kv_head, :], pos


def _kv_slice_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    page_table: torch.Tensor,
    batch_idx: int,
    kv_head: int,
    kv_block: int,
    blk_kv: int,
    cu_seqlens_k: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_len = cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx]
    local_start = kv_block * blk_kv
    local_end = min(local_start + blk_kv, k_len)
    if local_start >= local_end:
        empty = torch.empty(0, k.shape[-1], device=k.device, dtype=k.dtype)
        pos = torch.empty(0, device=k.device, dtype=torch.int32)
        return empty, empty.to(v.dtype), pos
    physical = int(page_table[batch_idx, kv_block].detach().cpu().item())
    page_len = local_end - local_start
    pos = torch.arange(local_start, local_end, device=k.device, dtype=torch.int32)
    return k[physical, kv_head, :page_len, :], v[physical, kv_head, :page_len, :], pos


def sparse_attention_csr_varlen_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    *,
    q2k_indices: Optional[torch.Tensor] = None,
    topk: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    lse_temperature_scale: float,
    return_temperature_lse: bool,
    return_softmax_lse: bool,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute public sparse attention semantics using portable PyTorch ops."""

    total_q, head_q, dim = q.shape
    head_kv = k.shape[1]
    if head_q % head_kv != 0:
        raise ValueError("q.shape[1] must be divisible by head_kv")
    if dim != 128:
        raise ValueError(f"SM120 reference currently expects D=128, got {dim}")
    if seqused_k is not None:
        # The SM100 public API supports paged effective lengths.  Keep the
        # fallback strict until a fused path handles this without host logic.
        cu_k = _to_int_list(cu_seqlens_k)
        used = _to_int_list(seqused_k)
        cu_k = [cu_k[0]] + [cu_k[i] + used[i] for i in range(len(used))]
    else:
        cu_k = _to_int_list(cu_seqlens_k)

    q2k = (
        q2k_indices.contiguous()
        if q2k_indices is not None
        else _reconstruct_q2k_from_k2q_csr(
            k2q_row_ptr,
            k2q_q_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            topk=topk,
            blk_kv=blk_kv,
            total_q=total_q,
        )
    )
    q_lookup = _query_batch_lookup(cu_seqlens_q)
    qhead_per_kv = head_q // head_kv
    scale = float(softmax_scale) if softmax_scale is not None else (dim ** -0.5)
    temp_inv = 1.0 / float(lse_temperature_scale)

    out_rows: list[torch.Tensor] = []
    lse_rows: list[torch.Tensor] = []
    temp_lse_rows: list[torch.Tensor] = []
    q2k_cpu = q2k.detach().cpu()

    for q_global, (batch_idx, q_local) in enumerate(q_lookup):
        head_out: list[torch.Tensor] = []
        head_lse: list[torch.Tensor] = []
        head_temp_lse: list[torch.Tensor] = []
        q_len = int(cu_seqlens_q[batch_idx + 1].item() - cu_seqlens_q[batch_idx].item())
        k_len = cu_k[batch_idx + 1] - cu_k[batch_idx]
        causal_limit = q_local + (k_len - q_len)
        for q_head in range(head_q):
            kv_head = q_head // qhead_per_kv
            k_parts: list[torch.Tensor] = []
            v_parts: list[torch.Tensor] = []
            pos_parts: list[torch.Tensor] = []
            for slot in range(topk):
                kv_block = int(q2k_cpu[kv_head, q_global, slot].item())
                if kv_block < 0:
                    continue
                if page_table is None:
                    kk, vv, pos = _kv_slice_dense(
                        k,
                        v,
                        batch_idx=batch_idx,
                        kv_head=kv_head,
                        kv_block=kv_block,
                        blk_kv=blk_kv,
                        cu_seqlens_k=cu_k,
                    )
                else:
                    kk, vv, pos = _kv_slice_paged(
                        k,
                        v,
                        page_table=page_table,
                        batch_idx=batch_idx,
                        kv_head=kv_head,
                        kv_block=kv_block,
                        blk_kv=blk_kv,
                        cu_seqlens_k=cu_k,
                    )
                if kk.numel() == 0:
                    continue
                k_parts.append(kk)
                v_parts.append(vv)
                pos_parts.append(pos)

            if not k_parts:
                head_out.append(torch.zeros(dim, dtype=torch.float32, device=q.device))
                neg_inf = torch.full((), float("-inf"), dtype=torch.float32, device=q.device)
                head_lse.append(neg_inf)
                if return_temperature_lse:
                    head_temp_lse.append(neg_inf)
                continue
            k_tokens = torch.cat(k_parts, dim=0).float()
            v_tokens = torch.cat(v_parts, dim=0).float()
            positions = torch.cat(pos_parts, dim=0)
            logits = torch.matmul(k_tokens, q[q_global, q_head, :].float()) * scale
            if causal:
                logits = logits.masked_fill(positions > causal_limit, float("-inf"))
            finite = torch.isfinite(logits)
            if not bool(finite.any().detach().cpu().item()):
                head_out.append(torch.zeros(dim, dtype=torch.float32, device=q.device))
                neg_inf = torch.full((), float("-inf"), dtype=torch.float32, device=q.device)
                head_lse.append(neg_inf)
                if return_temperature_lse:
                    head_temp_lse.append(neg_inf)
                continue
            lse = torch.logsumexp(logits, dim=0)
            probs = torch.softmax(logits, dim=0)
            head_out.append(torch.matmul(probs, v_tokens))
            head_lse.append(lse)
            if return_temperature_lse:
                head_temp_lse.append(torch.logsumexp(logits * temp_inv, dim=0))

        out_rows.append(torch.stack(head_out, dim=0))
        lse_rows.append(torch.stack(head_lse, dim=0))
        if return_temperature_lse:
            temp_lse_rows.append(torch.stack(head_temp_lse, dim=0))

    out = torch.stack(out_rows, dim=0).to(torch.bfloat16)
    lse_out = torch.stack(lse_rows, dim=0).to(torch.float32)
    if return_softmax_lse:
        if return_temperature_lse:
            return out, lse_out, torch.stack(temp_lse_rows, dim=0).to(torch.float32)
        return out, lse_out
    return out


def sparse_decode_paged_fp8_torch(
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
    quantize_p_fp8: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Portable paged FP8 decode reference for SM120 validation."""

    del max_seqlen_k
    if not causal:
        raise NotImplementedError("decode reference currently supports causal=True only")
    total_q, head_q, dim = q.shape
    head_kv = k.shape[1]
    qhead_per_kv = head_q // head_kv
    batch = int(page_table.shape[0])
    max_pages = int(page_table.shape[1])
    if total_q != batch * int(seqlen_q):
        raise ValueError("q.shape[0] must equal batch * seqlen_q")
    scale = float(softmax_scale) if softmax_scale is not None else (dim ** -0.5)
    q2k_cpu = None if q2k_indices is None else q2k_indices.detach().cpu()

    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)

    for batch_idx in range(batch):
        used_k = int(seqused_k[batch_idx].detach().cpu().item())
        q_begin = batch_idx * int(seqlen_q)
        for q_local in range(int(seqlen_q)):
            q_global = q_begin + q_local
            causal_limit = q_local + (used_k - int(seqlen_q))
            for q_head in range(head_q):
                kv_head = q_head // qhead_per_kv
                page_ids: list[int] = []
                if q2k_cpu is None:
                    page_ids = list(range(max_pages))
                else:
                    topk = int(q2k_indices.shape[-1])
                    for slot in range(topk):
                        page_id = int(q2k_cpu[kv_head, q_global, slot].item())
                        if page_id >= 0:
                            page_ids.append(page_id)
                k_parts: list[torch.Tensor] = []
                v_parts: list[torch.Tensor] = []
                pos_parts: list[torch.Tensor] = []
                for logical_page in page_ids:
                    if logical_page < 0 or logical_page >= max_pages:
                        continue
                    start = logical_page * int(blk_kv)
                    end = min(start + int(blk_kv), used_k)
                    if start >= end:
                        continue
                    physical_page = int(page_table[batch_idx, logical_page].detach().cpu().item())
                    page_len = end - start
                    k_parts.append(k[physical_page, kv_head, :page_len, :].float())
                    v_parts.append(v[physical_page, kv_head, :page_len, :].float())
                    pos_parts.append(torch.arange(start, end, device=q.device, dtype=torch.int32))
                if not k_parts:
                    out[q_global, q_head, :].zero_()
                    lse[q_global, q_head] = -float("inf")
                    continue
                k_tokens = torch.cat(k_parts, dim=0)
                v_tokens = torch.cat(v_parts, dim=0)
                positions = torch.cat(pos_parts, dim=0)
                logits = torch.matmul(k_tokens, q[q_global, q_head, :].float()) * scale
                logits = logits.masked_fill(positions > causal_limit, float("-inf"))
                finite = torch.isfinite(logits)
                if not bool(finite.any().detach().cpu().item()):
                    out[q_global, q_head, :].zero_()
                    lse[q_global, q_head] = -float("inf")
                    continue
                row_max = torch.max(logits)
                p = torch.exp(logits - row_max)
                p = torch.where(finite, p, torch.zeros_like(p))
                row_sum = p.sum()
                p_pv = p.to(torch.float8_e4m3fn).float() if quantize_p_fp8 else p
                out[q_global, q_head, :] = torch.matmul(p_pv, v_tokens) / row_sum
                lse[q_global, q_head] = row_max + torch.log(row_sum)

    out_bf16 = out.to(torch.bfloat16)
    if return_softmax_lse:
        return out_bf16, lse
    return out_bf16
