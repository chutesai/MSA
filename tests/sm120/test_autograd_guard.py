#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Verify SM120 sparse attention does not silently cut autograd."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from interface import sparse_atten_func  # noqa: E402
from sparse_index_utils import build_k2q_csr  # noqa: E402


def test_gradients_flow_through_triton_backend() -> None:
    os.environ["FMHA_SM120_BACKEND"] = "triton"
    seq = 512
    head_kv = 1
    qhead_per_kv = 2
    head_q = head_kv * qhead_per_kv
    dim = 128
    topk = 4
    blk_kv = 128
    q = torch.randn(seq, head_q, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(seq, head_kv, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(seq, head_kv, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q2k = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .view(1, 1, topk)
        .expand(head_kv, seq, topk)
        .contiguous()
    )
    cu = torch.tensor([0, seq], device="cuda", dtype=torch.int32)
    row, idx, schedule = build_k2q_csr(
        q2k,
        cu,
        cu,
        blk_kv,
        total_k=seq,
        max_seqlen_k=seq,
        max_seqlen_q=seq,
        total_rows=topk,
        qhead_per_kv=qhead_per_kv,
        return_schedule=True,
    )
    out = sparse_atten_func(
        q,
        k,
        v,
        row,
        idx,
        topk,
        blk_kv=blk_kv,
        causal=True,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        schedule=schedule,
        q2k_indices=q2k,
    )
    loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        grad = tensor.grad
        assert grad is not None, f"{name}.grad is None"
        assert torch.isfinite(grad.float()).all(), f"{name}.grad has non-finite values"
        assert float(grad.float().abs().sum().item()) > 0.0, f"{name}.grad is zero"
    print("autograd_guard: ok", float(loss.item()))


def main() -> int:
    test_gradients_flow_through_triton_backend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
