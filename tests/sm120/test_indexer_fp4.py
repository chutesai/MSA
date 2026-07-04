#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""SM120 Triton FP4 indexer vs the exact torch reference from the cute tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from src.sm120.fp4_indexer import fp4_indexer_block_scores_triton  # noqa: E402
from test_fp4_indexer import (  # noqa: E402  (cute/test_fp4_indexer.py)
    _make_random_score_case,
    _reference_block_scores,
)


def run_case(*, batch, max_seqlen, heads_q, heads_k, causal, seed, with_qo_offset=False):
    case = _make_random_score_case(
        fmt="mxfp4", batch=batch, max_seqlen=max_seqlen,
        heads_q=heads_q, heads_k=heads_k, seed=seed,
    )
    qo_offset = None
    if with_qo_offset:
        k_lens = case["cu_seqlens_k"].diff()
        q_lens = case["cu_seqlens_q"].diff()
        qo_offset = (k_lens - q_lens).to(torch.int32).contiguous()
    ref = _reference_block_scores(
        case["q"], case["k"], case["q_scale"], case["k_scale"],
        case["cu_seqlens_q"], case["cu_seqlens_k"], case["cu_page_offsets"],
        fmt="mxfp4", kv_indices=case["kv_indices"], causal=causal, qo_offset=qo_offset,
    ).cuda()
    got = fp4_indexer_block_scores_triton(
        case["q"], case["k"], case["q_scale"], case["k_scale"],
        case["cu_seqlens_q"], case["cu_seqlens_k"], case["cu_page_offsets"],
        max_seqlen_q=case["max_seqlen"], max_seqlen_k=case["max_seqlen"],
        kv_indices=case["kv_indices"], fp4_format="mxfp4",
        causal=causal, qo_offset=qo_offset,
    )
    torch.cuda.synchronize()
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert torch.equal(torch.isinf(got), torch.isinf(ref)), "-inf structure differs"
    finite = ~torch.isinf(ref)
    torch.testing.assert_close(got[finite], ref[finite], rtol=1e-3, atol=1e-3)
    print(f"ok fp4_indexer batch={batch} seq={max_seqlen} hq={heads_q} "
          f"causal={causal} qo_offset={with_qo_offset}")


def test_fp4_indexer_matches_reference() -> None:
    run_case(batch=2, max_seqlen=384, heads_q=4, heads_k=2, causal=False, seed=71)
    run_case(batch=2, max_seqlen=384, heads_q=4, heads_k=2, causal=True, seed=72)
    run_case(batch=3, max_seqlen=300, heads_q=8, heads_k=2, causal=True, seed=73)


def test_fp4_indexer_qo_offset() -> None:
    run_case(batch=2, max_seqlen=384, heads_q=4, heads_k=2, causal=True,
             seed=74, with_qo_offset=True)


def test_fp4_indexer_feeds_topk_shapes() -> None:
    """Output shape/dtype must be directly consumable by sparse_topk_select."""
    case = _make_random_score_case(
        fmt="mxfp4", batch=2, max_seqlen=384, heads_q=4, heads_k=2, seed=75,
    )
    got = fp4_indexer_block_scores_triton(
        case["q"], case["k"], case["q_scale"], case["k_scale"],
        case["cu_seqlens_q"], case["cu_seqlens_k"], case["cu_page_offsets"],
        max_seqlen_q=case["max_seqlen"], max_seqlen_k=case["max_seqlen"],
        kv_indices=case["kv_indices"], fp4_format="mxfp4", causal=True,
    )
    heads_q = int(case["q"].shape[1])
    total_q = int(case["q"].shape[0])
    assert got.shape == (heads_q, (case["max_seqlen"] + 127) // 128, total_q)
    assert got.dtype == torch.float32 and got.is_contiguous()
    print("ok fp4_indexer topk-compatible output")


def main() -> int:
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    test_fp4_indexer_matches_reference()
    test_fp4_indexer_qo_offset()
    test_fp4_indexer_feeds_topk_shapes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
