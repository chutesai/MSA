#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Compare the SM120 Triton sparse forward backend against torch reference."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from interface import sparse_atten_func  # noqa: E402
from sparse_index_utils import build_k2q_csr  # noqa: E402


def _make_q2k(
    head_kv: int,
    batch_lens: tuple[int, ...],
    topk: int,
    *,
    blk_kv: int,
    device: str,
) -> torch.Tensor:
    total_q = sum(batch_lens)
    rows = []
    for _h in range(head_kv):
        choices = []
        for length in batch_lens:
            num_blocks = (int(length) + blk_kv - 1) // blk_kv
            for _q in range(int(length)):
                perm = torch.randperm(num_blocks, device=device, dtype=torch.int32)
                if num_blocks >= topk:
                    choices.append(perm[:topk])
                    continue
                pad = torch.full((topk - num_blocks,), -1, device=device, dtype=torch.int32)
                choices.append(torch.cat([perm, pad], dim=0))
            if int(length) == 0:
                continue
        if not choices:
            rows.append(torch.full((total_q, topk), -1, device=device, dtype=torch.int32))
        else:
            rows.append(torch.stack(choices, dim=0))
    q2k = torch.stack(rows, dim=0).contiguous()
    num_blocks = max((int(length) + blk_kv - 1) // blk_kv for length in batch_lens)
    q2k_sort_key = torch.where(q2k < 0, torch.full_like(q2k, num_blocks), q2k)
    _, order = q2k_sort_key.sort(dim=-1)
    return q2k.gather(-1, order).contiguous()


def _pack_identity_pages(
    k: torch.Tensor,
    v: torch.Tensor,
    batch_lens: tuple[int, ...],
    *,
    blk_kv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_pages = sum((int(length) + blk_kv - 1) // blk_kv for length in batch_lens)
    max_pages = max((int(length) + blk_kv - 1) // blk_kv for length in batch_lens)
    head_kv = k.shape[1]
    dim = k.shape[2]
    k_pages = torch.zeros(total_pages, head_kv, blk_kv, dim, device=k.device, dtype=k.dtype)
    v_pages = torch.zeros_like(k_pages)
    page_table = torch.full((len(batch_lens), max_pages), -1, device=k.device, dtype=torch.int32)
    token_cursor = 0
    page_cursor = 0
    for batch_idx, length in enumerate(batch_lens):
        pages = (int(length) + blk_kv - 1) // blk_kv
        for page in range(pages):
            page_table[batch_idx, page] = page_cursor
            start = token_cursor + page * blk_kv
            end = min(start + blk_kv, token_cursor + int(length))
            page_len = end - start
            k_pages[page_cursor, :, :page_len, :] = k[start:end].transpose(0, 1)
            v_pages[page_cursor, :, :page_len, :] = v[start:end].transpose(0, 1)
            page_cursor += 1
        token_cursor += int(length)
    return k_pages.contiguous(), v_pages.contiguous(), page_table.contiguous()


def run_case(
    *,
    topk: int,
    causal: bool,
    batch_lens: tuple[int, ...],
    seed: int,
    paged: bool = False,
    fp8_kv: bool = False,
) -> None:
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    blk_kv = 128
    dim = 128
    head_kv = 2
    qhead_per_kv = 4
    head_q = head_kv * qhead_per_kv
    cu = [0]
    for length in batch_lens:
        cu.append(cu[-1] + int(length))
    cu_q = torch.tensor(cu, device=device, dtype=torch.int32)
    cu_k = torch.tensor(cu, device=device, dtype=torch.int32)
    total_q = cu[-1]
    total_k = cu[-1]
    max_len = max(batch_lens)
    total_rows = sum((length + blk_kv - 1) // blk_kv for length in batch_lens)
    q = torch.randn(total_q, head_q, dim, device=device, dtype=dtype)
    k = torch.randn(total_k, head_kv, dim, device=device, dtype=dtype)
    v = torch.randn(total_k, head_kv, dim, device=device, dtype=dtype)
    if fp8_kv:
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
    page_table = None
    k_call = k
    v_call = v
    if paged:
        k_call, v_call, page_table = _pack_identity_pages(k, v, batch_lens, blk_kv=blk_kv)
    q2k = _make_q2k(head_kv, batch_lens, topk, blk_kv=blk_kv, device=device)
    row, idx, schedule = build_k2q_csr(
        q2k,
        cu_q,
        cu_k,
        blk_kv,
        total_k=total_k,
        max_seqlen_k=max_len,
        max_seqlen_q=max_len,
        total_rows=total_rows,
        qhead_per_kv=qhead_per_kv,
        return_schedule=True,
    )
    kwargs = dict(
        blk_kv=blk_kv,
        causal=causal,
        return_softmax_lse=True,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_len,
        max_seqlen_k=max_len,
        schedule=schedule,
        q2k_indices=q2k,
    )
    old_backend = os.environ.get("FMHA_SM120_BACKEND")
    try:
        os.environ["FMHA_SM120_BACKEND"] = "torch_ref"
        ref_out, ref_lse = sparse_atten_func(
            q,
            k_call,
            v_call,
            row,
            idx,
            topk,
            page_table=page_table,
            **kwargs,
        )
        os.environ["FMHA_SM120_BACKEND"] = "triton"
        os.environ["FMHA_SM120_TRITON_STRICT"] = "1"
        tri_out, tri_lse = sparse_atten_func(
            q,
            k_call,
            v_call,
            row,
            idx,
            topk,
            page_table=page_table,
            **kwargs,
        )
    finally:
        if old_backend is None:
            os.environ.pop("FMHA_SM120_BACKEND", None)
        else:
            os.environ["FMHA_SM120_BACKEND"] = old_backend
    torch.cuda.synchronize()
    if fp8_kv:
        torch.testing.assert_close(tri_lse, ref_lse, rtol=4e-3, atol=8e-3)
        torch.testing.assert_close(tri_out, ref_out, rtol=3e-2, atol=3e-2)
    else:
        torch.testing.assert_close(tri_lse, ref_lse, rtol=2e-3, atol=3e-3)
        torch.testing.assert_close(tri_out, ref_out, rtol=2e-2, atol=2e-2)
    print(
        f"ok topk={topk} causal={causal} batch_lens={batch_lens} "
        f"paged={paged} fp8_kv={fp8_kv} out={tuple(tri_out.shape)}"
    )


def test_forward_bf16() -> None:
    for seed, topk, causal, lens in [
        (1, 4, False, (256,)),
        (2, 4, True, (256,)),
        (3, 16, False, (512,)),
        (4, 16, True, (384, 512)),
    ]:
        run_case(topk=topk, causal=causal, batch_lens=lens, seed=seed)


def test_forward_fp8_kv() -> None:
    run_case(topk=16, causal=True, batch_lens=(384, 512), seed=14, fp8_kv=True)


def test_forward_paged() -> None:
    run_case(topk=4, causal=True, batch_lens=(256, 384), seed=5, paged=True)
    run_case(topk=4, causal=True, batch_lens=(256, 384), seed=15, paged=True, fp8_kv=True)


def main() -> int:
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    test_forward_bf16()
    test_forward_fp8_kv()
    test_forward_paged()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
