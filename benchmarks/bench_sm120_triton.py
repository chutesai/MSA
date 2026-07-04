#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Microbenchmark the SM120 Triton sparse prefill backend."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from interface import sparse_atten_func  # noqa: E402
from sparse_index_utils import build_k2q_csr  # noqa: E402


def build_inputs(args):
    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16
    total_q = args.seq
    total_k = args.seq
    head_kv = args.head_kv
    head_q = args.head_kv * args.qhead_per_kv
    dim = 128
    blk_kv = args.blk_kv
    num_blocks = (total_k + blk_kv - 1) // blk_kv
    if args.topk > num_blocks:
        raise ValueError(f"topk={args.topk} exceeds num_blocks={num_blocks}")
    q = torch.randn(total_q, head_q, dim, device=device, dtype=dtype)
    k = torch.randn(total_k, head_kv, dim, device=device, dtype=dtype)
    v = torch.randn(total_k, head_kv, dim, device=device, dtype=dtype)
    if getattr(args, "fp8_kv", False):
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)
    if args.pattern == "prefix":
        base = torch.arange(args.topk, device=device, dtype=torch.int32)
        q2k = base.view(1, 1, args.topk).expand(head_kv, total_q, args.topk).contiguous()
    elif args.pattern == "random":
        rows = []
        for _ in range(head_kv):
            choices = [
                torch.randperm(num_blocks, device=device, dtype=torch.int32)[: args.topk]
                for _ in range(total_q)
            ]
            rows.append(torch.stack(choices, dim=0))
        q2k = torch.stack(rows, dim=0).contiguous()
        q2k_sort_key = torch.where(q2k < 0, torch.full_like(q2k, num_blocks), q2k)
        _, order = q2k_sort_key.sort(dim=-1)
        q2k = q2k.gather(-1, order).contiguous()
    elif args.pattern == "local_sink":
        # Sink block 0 plus the (topk-1) blocks ending at each token's own
        # block: neighboring tokens share almost all selections, which is the
        # regime real MSA top-k selection produces.
        pos = torch.arange(total_q, device=device)
        own = (pos // blk_kv).to(torch.int32)
        offs = torch.arange(args.topk - 1, device=device, dtype=torch.int32).flip(0)
        local = own.unsqueeze(-1) - offs.unsqueeze(0)
        q2k = torch.cat(
            [torch.zeros(total_q, 1, device=device, dtype=torch.int32), local], dim=1
        )
        q2k = torch.where(q2k < 0, -1, q2k).sort(dim=-1).values
        dup = q2k[:, 1:] == q2k[:, :-1]
        q2k[:, 1:][dup] = -1
        key = torch.where(q2k < 0, num_blocks, q2k)
        q2k = q2k.gather(-1, key.argsort(dim=-1))
        q2k = q2k.unsqueeze(0).expand(head_kv, total_q, args.topk).contiguous()
    else:
        raise ValueError(args.pattern)
    cu = torch.tensor([0, total_q], device=device, dtype=torch.int32)
    row, idx, schedule = build_k2q_csr(
        q2k,
        cu,
        cu,
        blk_kv,
        total_k=total_k,
        max_seqlen_k=total_k,
        max_seqlen_q=total_q,
        total_rows=num_blocks,
        qhead_per_kv=args.qhead_per_kv,
        return_schedule=True,
    )
    return q, k, v, row, idx, schedule, cu, q2k


def time_backend(args, backend: str, mode: str) -> tuple[float, int, int]:
    q, k, v, row, idx, schedule, cu, q2k = build_inputs(args)
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
    os.environ["FMHA_SM120_BACKEND"] = backend
    os.environ["FMHA_SM120_TRITON_STRICT"] = "1" if backend == "triton" else "0"
    os.environ["FMHA_SM120_TRITON_MODE"] = mode
    os.environ["FMHA_SM120_Q_CHUNK"] = str(args.q_chunk)
    os.environ["FMHA_SM120_MAX_PARTIAL_MIB"] = str(args.max_partial_mib)
    for _ in range(args.warmup):
        sparse_atten_func(q, k, v, row, idx, args.topk, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        sparse_atten_func(q, k, v, row, idx, args.topk, **kwargs)
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
    parser.add_argument("--pattern", choices=("prefix", "random", "local_sink"), default="random")
    parser.add_argument("--fp8-kv", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", default=None, help="Single FMHA_SM120_TRITON_MODE to run.")
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated modes to run, e.g. two_phase,chunked,row.",
    )
    parser.add_argument("--q-chunk", type=int, default=4096)
    parser.add_argument("--max-partial-mib", type=int, default=1024)
    args = parser.parse_args()
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    modes = (
        [m.strip() for m in args.modes.split(",") if m.strip()]
        if args.modes
        else [args.mode or os.environ.get("FMHA_SM120_TRITON_MODE", "auto")]
    )
    for mode in modes:
        tri_ms, peak_alloc, peak_reserved = time_backend(args, "triton", mode)
        print(
            f"triton mode={mode} ms={tri_ms:.3f} "
            f"peak_alloc_mib={peak_alloc / (1024 ** 2):.1f} "
            f"peak_reserved_mib={peak_reserved / (1024 ** 2):.1f}"
        )
    if args.seq <= 1024:
        ref_ms, ref_alloc, ref_reserved = time_backend(args, "torch_ref", "reference")
        print(
            f"torch_ref_ms {ref_ms:.3f} "
            f"peak_alloc_mib={ref_alloc / (1024 ** 2):.1f} "
            f"peak_reserved_mib={ref_reserved / (1024 ** 2):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
