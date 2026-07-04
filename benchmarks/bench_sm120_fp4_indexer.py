#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Benchmark the SM120 FP4 indexer against a BF16 cuBLAS scoring baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from src.sm120.fp4_indexer import fp4_indexer_block_scores_triton  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--heads-q", type=int, default=16)
    parser.add_argument("--heads-k", type=int, default=4)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    seq, hq, hk = args.seq, args.heads_q, args.heads_k
    pages = (seq + 127) // 128
    q4 = torch.randint(0, 256, (seq, hq, 64), dtype=torch.uint8, device=dev)
    k4 = torch.randint(0, 256, (pages, hk, 128, 64), dtype=torch.uint8, device=dev)
    # ue8m0 near 1.0 keeps scores in a realistic range
    qs = torch.randint(124, 131, (seq, hq, 4), dtype=torch.uint8, device=dev)
    ks = torch.randint(124, 131, (pages, hk, 128, 4), dtype=torch.uint8, device=dev)
    cu_q = torch.tensor([0, seq], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seq], dtype=torch.int32, device=dev)
    cu_p = torch.tensor([0, pages], dtype=torch.int32, device=dev)
    kv_idx = torch.arange(pages, dtype=torch.int32, device=dev)

    def run_fp4():
        return fp4_indexer_block_scores_triton(
            q4, k4, qs, ks, cu_q, cu_k, cu_p,
            max_seqlen_q=seq, max_seqlen_k=seq, kv_indices=kv_idx,
            fp4_format="mxfp4", causal=args.causal,
        )

    # BF16 baseline: dense scores via cuBLAS + page max (what scoring costs
    # without the FP4 path).
    qb = torch.randn(hq, seq, 128, device=dev, dtype=torch.bfloat16)
    kb = torch.randn(hk, seq, 128, device=dev, dtype=torch.bfloat16)
    kb_g = kb.repeat_interleave(hq // hk, dim=0)

    def run_bf16():
        s = torch.bmm(qb, kb_g.transpose(1, 2))  # (hq, seq, seq)
        return s.view(hq, seq, pages, 128).amax(dim=-1)

    def timeit(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / args.iters

    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    flops = 2.0 * seq * seq * 128 * hq * (0.5 if args.causal else 1.0)
    ms = timeit(run_fp4)
    print(f"fp4_indexer  causal={args.causal} ms={ms:.3f} eff_tflops={flops / ms / 1e9:.1f}")
    ms = timeit(run_bf16)
    print(f"bf16_baseline (dense, non-causal) ms={ms:.3f} tflops={2.0 * seq * seq * 128 * hq / ms / 1e9:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
