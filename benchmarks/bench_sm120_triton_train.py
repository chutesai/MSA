#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Benchmark the SM120 Triton sparse prefill training path.

This times the public sparse_atten_func autograd route:
    forward -> synthetic upstream gradient -> backward

The forward-only benchmark is useful for decode/prefill serving work, but the
training path is dominated by the backward kernel and saved/intermediate state.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from bench_sm120_triton import build_inputs  # noqa: E402
from interface import sparse_atten_func  # noqa: E402


def _run_one(args, mode: str) -> tuple[float, int, int]:
    os.environ["FMHA_SM120_BACKEND"] = "triton"
    os.environ["FMHA_SM120_TRITON_STRICT"] = "1"
    os.environ["FMHA_SM120_BACKWARD"] = "triton"
    os.environ["FMHA_SM120_TRITON_MODE"] = mode
    os.environ["FMHA_SM120_Q_CHUNK"] = str(args.q_chunk)
    os.environ["FMHA_SM120_MAX_PARTIAL_MIB"] = str(args.max_partial_mib)
    os.environ["FMHA_SM120_PARTIAL_DTYPE"] = args.partial_dtype

    q, k, v, row, idx, schedule, cu, q2k = build_inputs(args)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    dout = torch.randn_like(q)
    kwargs = dict(
        blk_kv=args.blk_kv,
        causal=args.causal,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=args.seq,
        max_seqlen_k=args.seq,
        schedule=schedule,
        q2k_indices=q2k,
    )

    def step() -> None:
        q.grad = None
        k.grad = None
        v.grad = None
        out = sparse_atten_func(q, k, v, row, idx, args.topk, **kwargs)
        out.backward(dout)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        step()
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
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--modes", default="two_phase,recompute,row")
    parser.add_argument("--q-chunk", type=int, default=4096)
    parser.add_argument("--max-partial-mib", type=int, default=1024)
    parser.add_argument("--partial-dtype", choices=("bf16", "fp32"), default="fp32")
    args = parser.parse_args()

    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        ms, peak_alloc, peak_reserved = _run_one(args, mode)
        print(
            f"triton_train mode={mode} ms={ms:.3f} "
            f"peak_alloc_mib={peak_alloc / (1024 ** 2):.1f} "
            f"peak_reserved_mib={peak_reserved / (1024 ** 2):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
