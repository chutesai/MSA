#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Benchmark SM120 packed NVFP4 K/V sparse prefill."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from interface import sparse_atten_nvfp4_kv_func  # noqa: E402
from sparse_index_utils import build_k2q_csr  # noqa: E402


def _make_scales(required_rows: int, *, device: str) -> torch.Tensor:
    padded_rows = ((int(required_rows) + 127) // 128) * 128
    padded_cols = 8
    # E4M3FN encoding of +1.0: sign=0, exp=bias=7, mant=0.
    return torch.full((padded_rows, padded_cols), 0x38, dtype=torch.uint8, device=device)


def _pack_from_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    low = nibbles[..., 0::2]
    high = nibbles[..., 1::2]
    return (low | (high << 4)).contiguous().to(torch.uint8)


def _make_q2k(head_kv: int, seq: int, topk: int, *, blk_kv: int, device: str) -> torch.Tensor:
    num_blocks = (seq + blk_kv - 1) // blk_kv
    rows = []
    for _ in range(head_kv):
        choices = [
            torch.randperm(num_blocks, device=device, dtype=torch.int32)[:topk]
            for _ in range(seq)
        ]
        rows.append(torch.stack(choices, dim=0))
    q2k = torch.stack(rows, dim=0).contiguous()
    q2k_sort_key = torch.where(q2k < 0, torch.full_like(q2k, num_blocks), q2k)
    _, order = q2k_sort_key.sort(dim=-1)
    return q2k.gather(-1, order).contiguous()


def _build_inputs(args):
    torch.manual_seed(args.seed)
    device = "cuda"
    dim = 128
    total_q = args.seq
    head_kv = args.head_kv
    head_q = head_kv * args.qhead_per_kv
    q = torch.randn(total_q, head_q, dim, device=device, dtype=torch.bfloat16)
    k_nibbles = torch.randint(0, 16, (total_q, head_kv, dim), device=device, dtype=torch.uint8)
    v_nibbles = torch.randint(0, 16, (total_q, head_kv, dim), device=device, dtype=torch.uint8)
    k = _pack_from_nibbles(k_nibbles)
    v = _pack_from_nibbles(v_nibbles)
    k_scale = _make_scales(total_q * head_kv, device=device)
    v_scale = _make_scales(total_q * head_kv, device=device)
    # Small global scale keeps random FP4 logits in a realistic numeric range.
    k_global = torch.tensor([0.125], device=device, dtype=torch.float32)
    v_global = torch.tensor([0.125], device=device, dtype=torch.float32)
    q2k = _make_q2k(head_kv, total_q, args.topk, blk_kv=args.blk_kv, device=device)
    cu = torch.tensor([0, total_q], device=device, dtype=torch.int32)
    row, idx, schedule = build_k2q_csr(
        q2k,
        cu,
        cu,
        args.blk_kv,
        total_k=total_q,
        max_seqlen_k=total_q,
        max_seqlen_q=total_q,
        total_rows=(total_q + args.blk_kv - 1) // args.blk_kv,
        qhead_per_kv=args.qhead_per_kv,
        return_schedule=True,
    )
    return q, k, v, k_scale, v_scale, k_global, v_global, row, idx, schedule, cu, q2k


def _time(args, mode: str) -> tuple[float, int, int]:
    os.environ["FMHA_SM120_BACKEND"] = "triton"
    os.environ["FMHA_SM120_TRITON_STRICT"] = "1"
    os.environ["FMHA_SM120_NVFP4_MODE"] = mode
    q, k, v, ks, vs, kg, vg, row, idx, schedule, cu, q2k = _build_inputs(args)
    kwargs = dict(
        blk_kv=args.blk_kv,
        causal=args.causal,
        return_softmax_lse=True,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=args.seq,
        max_seqlen_k=args.seq,
        schedule=schedule,
        q2k_indices=q2k,
    )
    for _ in range(args.warmup):
        sparse_atten_nvfp4_kv_func(q, k, v, ks, vs, kg, vg, row, idx, args.topk, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        sparse_atten_nvfp4_kv_func(q, k, v, ks, vs, kg, vg, row, idx, args.topk, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return (
        start.elapsed_time(end) / args.iters,
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--head-kv", type=int, default=4)
    parser.add_argument("--qhead-per-kv", type=int, default=4)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--blk-kv", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--modes", default="csr,row")
    args = parser.parse_args()
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        ms, peak_alloc, peak_reserved = _time(args, mode)
        print(
            f"nvfp4 mode={mode} ms={ms:.3f} "
            f"peak_alloc_mib={peak_alloc / (1024 ** 2):.1f} "
            f"peak_reserved_mib={peak_reserved / (1024 ** 2):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
