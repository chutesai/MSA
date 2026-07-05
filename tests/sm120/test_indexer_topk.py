# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Fused FP4 indexer top-k vs the scores-kernel + torch selection chain."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python" / "fmha_sm100" / "cute"))

from src.sm120.fp4_indexer import (  # noqa: E402
    fp4_indexer_block_scores_triton,
    fp4_indexer_topk,
)
from src.sm120.qstat import quantize_mxfp4  # noqa: E402


def _make_case(batch, seq, hq, hk, seed):
    torch.manual_seed(seed)
    total = batch * seq
    pages = total // 128
    q = torch.randn(total, hq, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, hk, 128, device="cuda", dtype=torch.bfloat16)
    q4, qsc = quantize_mxfp4(q)
    k4_flat, ksc_flat = quantize_mxfp4(k)
    # identity paging: page p holds tokens [p*128, (p+1)*128)
    k4 = k4_flat.view(pages, 128, hk, 64).permute(0, 2, 1, 3).contiguous()
    ksc = ksc_flat.view(pages, 128, hk, 4).permute(0, 2, 1, 3).contiguous()
    cu_q = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32) * seq
    cu_pages = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32) * (seq // 128)
    kv_indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    return q4, k4, qsc, ksc, cu_q, cu_pages, kv_indices


def _reference_topk(scores, cu_q, seq, topk, force_diag):
    """torch selection chain over the scores kernel's output."""
    hq, tiles, total = scores.shape
    out = torch.full((hq, total, topk), -1, dtype=torch.int32, device=scores.device)
    for h in range(hq):
        s = scores[h].t().clone()  # [total_q, tiles]
        for qi in range(total):
            local = qi % seq
            row = s[qi]
            finite = torch.isfinite(row)
            if not finite.any():
                continue
            k_slots = topk
            picked = []
            if force_diag:
                diag = local // 128
                picked.append(diag)
                row = row.clone()
                row[diag] = float("-inf")
                k_slots -= 1
            vals, idx = row.topk(min(k_slots, int(finite.sum().item())))
            picked.extend(idx[torch.isfinite(vals)].tolist())
            picked = sorted(set(picked))
            out[h, qi, : len(picked)] = torch.tensor(
                picked, dtype=torch.int32, device=scores.device
            )
    return out


def _check(batch, seq, hq, hk, topk, seed, force_diag=1):
    q4, k4, qsc, ksc, cu_q, cu_pages, kv_indices = _make_case(batch, seq, hq, hk, seed)
    got = fp4_indexer_topk(
        q4, k4, qsc, ksc, cu_q, cu_q, cu_pages,
        topk=topk, max_seqlen_q=seq, max_seqlen_k=seq, kv_indices=kv_indices,
        causal=True, force_diagonal_blocks=force_diag,
    )
    scores = fp4_indexer_block_scores_triton(
        q4, k4, qsc, ksc, cu_q, cu_q, cu_pages,
        max_seqlen_q=seq, max_seqlen_k=seq, kv_indices=kv_indices,
        fp4_format="mxfp4", causal=True,
    )
    ref = _reference_topk(scores, cu_q, seq, topk, force_diag)
    total = batch * seq
    # Format: ascending valid ids, -1 tail.
    for h in range(hq):
        v = got[h]
        valid = v >= 0
        asc = torch.all((v[:, 1:] >= v[:, :-1]) | ~valid[:, 1:])
        assert bool(asc), "ids not ascending"
        tail_ok = torch.all((~valid[:, :-1]) <= (~valid[:, 1:]))
        assert bool(tail_ok), "-1 not confined to tail"
    # Selection quality: per row, the set of selected true scores must match
    # the reference's (tie-robust: compare summed selected scores).
    mism = 0
    for h in range(hq):
        sc = scores[h].t()  # [total, tiles]
        for qi in range(0, total, max(1, total // 512)):
            g = got[h, qi]
            r = ref[h, qi]
            gs = sc[qi][g[g >= 0].long()]
            rs = sc[qi][r[r >= 0].long()]
            assert len(gs) == len(rs), f"count mismatch h{h} q{qi}: {len(gs)} vs {len(rs)}"
            if force_diag:
                assert (qi % seq) // 128 in g[g >= 0].tolist(), "diagonal not forced"
            if not torch.isclose(gs.sum(), rs.sum(), rtol=1e-4, atol=1e-3):
                mism += 1
    assert mism == 0, f"{mism} rows selected different score mass"
    print(f"ok topk batch={batch} seq={seq} hq={hq} hk={hk} topk={topk} fd={force_diag}")


def test_indexer_topk_matches_reference():
    _check(1, 512, 2, 2, 4, seed=1)
    _check(1, 1024, 2, 1, 8, seed=2)
    _check(2, 512, 4, 2, 8, seed=3)
    _check(1, 1024, 2, 2, 16, seed=4)
    _check(1, 512, 2, 2, 4, seed=5, force_diag=0)


if __name__ == "__main__":
    test_indexer_topk_matches_reference()
    print("INDEXER_TOPK_OK")
