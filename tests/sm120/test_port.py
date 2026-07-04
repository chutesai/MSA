#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""SM120 port smoke test: csrc top-k select, CSR builder, sparse attention.

Run on RTX PRO 6000 Blackwell with:

    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:$PATH \
    FMHA_CUDA_ARCH=120 \
    CUTE_DSL_ARCH=sm_120 \
    python tests/sm120/test_port.py

The SM100 CuTe/tcgen05 kernels remain unsupported on sm_120; set
FMHA_SM120_BACKEND=off to reproduce the NVVM unsupported-operation failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))


def _print_env() -> None:
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("FMHA_CUDA_ARCH", os.environ.get("FMHA_CUDA_ARCH"))
    print("CUTE_DSL_ARCH", os.environ.get("CUTE_DSL_ARCH"))


def test_sparse_topk_select() -> None:
    from fmha_sm100 import sparse_topk_select

    heads, blocks, q_tokens = 2, 32, 7
    scores = torch.randn(heads, blocks, q_tokens, device="cuda", dtype=torch.float32).contiguous()
    out = sparse_topk_select(scores, 16, num_valid_pages=24, force_begin_blocks=1, force_end_blocks=1)
    torch.cuda.synchronize()
    assert out.shape == (q_tokens, heads, 16)
    assert out.dtype == torch.int32
    assert ((out == -1) | ((out >= 0) & (out < 24))).all()
    print("sparse_topk_select: ok", out[0, 0].tolist())


def test_csr_builder() -> None:
    from sparse_index_utils import build_k2q_csr

    seq_q = seq_k = 512
    heads_kv = 1
    topk = 4
    block_kv = 128
    q2k = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .view(1, 1, topk)
        .expand(heads_kv, seq_q, topk)
        .contiguous()
    )
    cu_q = torch.tensor([0, seq_q], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, seq_k], device="cuda", dtype=torch.int32)
    row, idx, schedule = build_k2q_csr(
        q2k,
        cu_q,
        cu_k,
        block_kv,
        total_k=seq_k,
        max_seqlen_k=seq_k,
        max_seqlen_q=seq_q,
        total_rows=(seq_k + block_kv - 1) // block_kv,
        qhead_per_kv=16,
        return_schedule=True,
    )
    torch.cuda.synchronize()
    print("build_k2q_csr: ok", row.shape, idx.shape, int(schedule.work_count.item()))


def test_sparse_attention_matches_dense_reference() -> None:
    from interface import sparse_atten_func
    from sparse_index_utils import build_k2q_csr

    seq_q = seq_k = 512
    heads_kv = 1
    qhead_per_kv = 16
    heads_q = heads_kv * qhead_per_kv
    dim = 128
    topk = 4
    block_kv = 128
    q = torch.randn(seq_q, heads_q, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq_k, heads_kv, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(seq_k, heads_kv, dim, device="cuda", dtype=torch.bfloat16)
    q2k = (
        torch.arange(topk, device="cuda", dtype=torch.int32)
        .view(1, 1, topk)
        .expand(heads_kv, seq_q, topk)
        .contiguous()
    )
    cu_q = torch.tensor([0, seq_q], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, seq_k], device="cuda", dtype=torch.int32)
    row, idx, schedule = build_k2q_csr(
        q2k,
        cu_q,
        cu_k,
        block_kv,
        total_k=seq_k,
        max_seqlen_k=seq_k,
        max_seqlen_q=seq_q,
        total_rows=topk,
        qhead_per_kv=qhead_per_kv,
        return_schedule=True,
    )
    out, lse = sparse_atten_func(
        q,
        k,
        v,
        row,
        idx,
        topk,
        blk_kv=block_kv,
        causal=False,
        return_softmax_lse=True,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=seq_q,
        max_seqlen_k=seq_k,
        schedule=schedule,
        q2k_indices=q2k,
    )
    torch.cuda.synchronize()
    # The sparse pattern covers every KV block, so a dense softmax is exact.
    k_rep = k.repeat_interleave(qhead_per_kv, dim=1)
    v_rep = v.repeat_interleave(qhead_per_kv, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k_rep.float()) / (dim ** 0.5)
    ref_lse = torch.logsumexp(scores, dim=-1).transpose(0, 1).contiguous()
    probs = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("hqk,khd->qhd", probs, v_rep.float()).to(torch.bfloat16)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)
    print("sparse_atten_func: ok", out.shape, lse.shape)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0
    if torch.cuda.get_device_properties(0).major < 12:
        print("SKIP: SM120-class GPU not available")
        return 0
    _print_env()
    test_sparse_topk_select()
    test_csr_builder()
    test_sparse_attention_matches_dense_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
