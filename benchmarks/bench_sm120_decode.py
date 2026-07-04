#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Benchmark SM120 paged FP8 decode attention."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from interface import sparse_decode_atten_func  # noqa: E402


def _make_inputs(*, batch: int, seqlen_q: int, kv_tokens: int, head_kv: int, seed: int):
    torch.manual_seed(seed)
    device = "cuda"
    blk_kv = 128
    dim = 128
    qhead_per_kv = 16
    head_q = head_kv * qhead_per_kv
    page_count = (kv_tokens + blk_kv - 1) // blk_kv
    q = torch.randn(batch * seqlen_q, head_q, dim, device=device).to(torch.float8_e4m3fn)
    k = torch.randn(batch, page_count, head_kv, blk_kv, dim, device=device).to(torch.float8_e4m3fn)
    v = torch.randn(batch, page_count, head_kv, blk_kv, dim, device=device).to(torch.float8_e4m3fn)
    tail = kv_tokens - (page_count - 1) * blk_kv
    if tail < blk_kv:
        k[:, -1, :, tail:, :] = 0
        v[:, -1, :, tail:, :] = 0
    k_pages = k.reshape(batch * page_count, head_kv, blk_kv, dim).contiguous()
    v_pages = v.reshape(batch * page_count, head_kv, blk_kv, dim).contiguous()
    page_table = torch.arange(batch * page_count, device=device, dtype=torch.int32).view(batch, page_count)
    seqused_k = torch.full((batch,), kv_tokens, dtype=torch.int32, device=device)
    return q, k_pages, v_pages, page_table, seqused_k


def _make_sparse_q2k(*, head_kv: int, total_q: int, page_count: int, topk: int, device: str) -> torch.Tensor:
    pages = torch.arange(page_count, device=device, dtype=torch.int32)
    if topk <= page_count:
        selected = pages[-topk:]
    else:
        selected = torch.cat(
            [pages, torch.full((topk - page_count,), -1, device=device, dtype=torch.int32)]
        )
    return selected.view(1, 1, topk).expand(head_kv, total_q, topk).contiguous()


def _bench(args, sparse: bool) -> tuple[float, int, int]:
    q, k, v, page_table, seqused_k = _make_inputs(
        batch=args.batch,
        seqlen_q=args.seqlen_q,
        kv_tokens=args.kv_tokens,
        head_kv=args.head_kv,
        seed=args.seed,
    )
    q2k = None
    if sparse:
        q2k = _make_sparse_q2k(
            head_kv=args.head_kv,
            total_q=q.shape[0],
            page_count=page_table.shape[1],
            topk=args.topk,
            device="cuda",
        )
    os.environ["FMHA_SM120_BACKEND"] = "triton"
    os.environ["FMHA_SM120_TRITON_STRICT"] = "1"
    os.environ["FMHA_SM120_DECODE_SPLIT_PAGES"] = str(args.split_pages)
    kwargs = dict(
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=args.seqlen_q,
        max_seqlen_k=args.kv_tokens,
        blk_kv=args.blk_kv,
        causal=True,
        return_softmax_lse=True,
    )
    for _ in range(args.warmup):
        sparse_decode_atten_func(q, k, v, q2k, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        sparse_decode_atten_func(q, k, v, q2k, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return (
        start.elapsed_time(end) / args.iters,
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seqlen-q", type=int, default=1)
    parser.add_argument("--kv-tokens", type=int, default=32768)
    parser.add_argument("--head-kv", type=int, default=2)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--blk-kv", type=int, default=128)
    parser.add_argument("--split-pages", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()

    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    ms, peak_alloc, peak_reserved = _bench(args, args.sparse)
    print(
        f"decode sparse={args.sparse} split_pages={args.split_pages} "
        f"batch={args.batch} seqlen_q={args.seqlen_q} kv_tokens={args.kv_tokens} "
        f"topk={args.topk} ms={ms:.3f} "
        f"peak_alloc_mib={peak_alloc / (1024 ** 2):.1f} "
        f"peak_reserved_mib={peak_reserved / (1024 ** 2):.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
