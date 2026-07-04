#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Compare SM120 Triton backward against the torch reference backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from interface import sparse_atten_func  # noqa: E402
from sparse_index_utils import build_k2q_csr  # noqa: E402


def _q2k(head_kv: int, seq: int, topk: int, *, blk_kv: int, device: str) -> torch.Tensor:
    num_blocks = (seq + blk_kv - 1) // blk_kv
    rows = []
    for _ in range(head_kv):
        per_q = []
        for _q in range(seq):
            if num_blocks >= topk:
                per_q.append(torch.randperm(num_blocks, device=device, dtype=torch.int32)[:topk])
            else:
                pad = torch.full((topk - num_blocks,), -1, device=device, dtype=torch.int32)
                per_q.append(torch.cat([torch.arange(num_blocks, device=device, dtype=torch.int32), pad], dim=0))
        rows.append(torch.stack(per_q, dim=0))
    q2k = torch.stack(rows, dim=0).contiguous()
    sort_key = torch.where(q2k < 0, torch.full_like(q2k, num_blocks), q2k)
    _, order = sort_key.sort(dim=-1)
    return q2k.gather(-1, order).contiguous()


def _run(backend: str, q, k, v, row, idx, q2k, schedule, cu, *, topk: int, causal: bool, dout: torch.Tensor):
    os.environ["FMHA_SM120_BACKEND"] = backend
    if backend == "triton":
        os.environ["FMHA_SM120_BACKWARD"] = "triton"
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = sparse_atten_func(
        q,
        k,
        v,
        row,
        idx,
        topk,
        blk_kv=128,
        causal=causal,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=q.shape[0],
        max_seqlen_k=k.shape[0],
        schedule=schedule,
        q2k_indices=q2k,
    )
    loss = (out.float() * dout.float()).sum()
    loss.backward()
    torch.cuda.synchronize()
    return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def run_case(*, seq: int, topk: int, causal: bool, seed: int) -> None:
    torch.manual_seed(seed)
    # Validate the backward formula with FP32 forward partials first.  BF16
    # partials are a performance mode and have their own forward tolerance.
    os.environ["FMHA_SM120_PARTIAL_DTYPE"] = "fp32"
    device = "cuda"
    dtype = torch.bfloat16
    head_kv = 2
    qhead_per_kv = 4
    head_q = head_kv * qhead_per_kv
    dim = 128
    q = torch.randn(seq, head_q, dim, device=device, dtype=dtype)
    k = torch.randn(seq, head_kv, dim, device=device, dtype=dtype)
    v = torch.randn(seq, head_kv, dim, device=device, dtype=dtype)
    dout = torch.randn(seq, head_q, dim, device=device, dtype=dtype)
    q2k = _q2k(head_kv, seq, topk, blk_kv=128, device=device)
    cu = torch.tensor([0, seq], device=device, dtype=torch.int32)
    total_rows = (seq + 127) // 128
    row, idx, schedule = build_k2q_csr(
        q2k,
        cu,
        cu,
        128,
        total_k=seq,
        max_seqlen_k=seq,
        max_seqlen_q=seq,
        total_rows=total_rows,
        qhead_per_kv=qhead_per_kv,
        return_schedule=True,
    )
    ref = _run("torch_ref", q, k, v, row, idx, q2k, schedule, cu, topk=topk, causal=causal, dout=dout)
    tri = _run("triton", q, k, v, row, idx, q2k, schedule, cu, topk=topk, causal=causal, dout=dout)
    names = ("out", "dq", "dk", "dv")
    for name, ref_t, tri_t in zip(names, ref, tri):
        # dk/dv use atomic accumulation in the Triton backend.  With BF16
        # inputs and BF16 gradient return, a handful of elements can differ by
        # one or two BF16 ulps versus PyTorch's deterministic reference path.
        if name in {"dk", "dv"}:
            torch.testing.assert_close(tri_t.float(), ref_t.float(), rtol=1e-1, atol=1e-1)
        else:
            torch.testing.assert_close(tri_t.float(), ref_t.float(), rtol=8e-2, atol=5e-2)
    max_diffs = {
        name: float((tri_t.float() - ref_t.float()).abs().max().item())
        for name, ref_t, tri_t in zip(names, ref, tri)
    }
    print(f"ok seq={seq} topk={topk} causal={causal} max_diffs={max_diffs}")


def test_backward_matches_reference() -> None:
    run_case(seq=256, topk=4, causal=False, seed=11)
    run_case(seq=256, topk=4, causal=True, seed=12)
    run_case(seq=512, topk=4, causal=True, seed=13)


def main() -> int:
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    test_backward_matches_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
