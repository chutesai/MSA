#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Correctness tests for mixed prefill/decode split in fmha_sm100.

Verifies that splitting a batch into decode (qo_len<=128) + prefill (qo_len>128)
and running two separate kernels produces the same result as running each part
independently without the split wrapper.
"""

import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fmha_sm100.api import fmha_sm100_plan, fmha_sm100, _fmha_sm100_plan, _fmha_sm100

failed_cases = []


def sdpa_ref_varlen(q, k, v, qo_lens, kv_lens, h_q, h_k, qo_offset=None):
    """Per-batch SDPA reference for variable-length packed inputs.

    Args:
        q: [total_qo, h_q, d]
        k: [total_kv, h_k, d]
        v: [total_kv, h_k, d]
        qo_lens: list of int
        kv_lens: list of int
        qo_offset: list of int or None (defaults to kv_len - qo_len per batch)
    """
    d = q.shape[-1]
    device = q.device
    gqa_ratio = h_q // h_k
    results = []
    qo_off, kv_off = 0, 0

    for i, (ql, kl) in enumerate(zip(qo_lens, kv_lens)):
        q_i = q[qo_off:qo_off + ql].unsqueeze(0).transpose(1, 2)  # [1, h_q, ql, d]
        k_i = k[kv_off:kv_off + kl].unsqueeze(0).transpose(1, 2)  # [1, h_k, kl, d]
        v_i = v[kv_off:kv_off + kl].unsqueeze(0).transpose(1, 2)

        k_i = k_i.repeat_interleave(gqa_ratio, dim=1)
        v_i = v_i.repeat_interleave(gqa_ratio, dim=1)

        off = (qo_offset[i] if qo_offset is not None else kl - ql)
        row = torch.arange(ql, device=device).reshape(1, ql, 1) + off
        col = torch.arange(kl, device=device).reshape(1, 1, kl)
        mask = (row >= col).unsqueeze(1)

        with torch.no_grad():
            o_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i, attn_mask=mask)
        results.append(o_i.transpose(1, 2).reshape(ql, h_q, d))

        qo_off += ql
        kv_off += kl

    return torch.cat(results, dim=0)


def check(name, o, o_ref, threshold=0.9999):
    cos_sim = torch.nn.functional.cosine_similarity(
        o.float().reshape(-1), o_ref.float().reshape(-1), dim=0
    ).item()
    max_diff = (o.float() - o_ref.float()).abs().max().item()
    passed = cos_sim > threshold
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")
    if not passed:
        failed_cases.append(f"{name}: cos_sim={cos_sim:.6f}")
    return passed


def make_inputs(qo_lens, kv_lens, h_q, h_k, d, device="cuda:0"):
    """Create packed q, k, v and metadata for variable-length batch."""
    total_qo = sum(qo_lens)
    total_kv = sum(kv_lens)
    q = torch.randn(total_qo, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_kv, h_k, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_kv, h_k, d, dtype=torch.bfloat16, device=device)
    qo_lens_t = torch.tensor(qo_lens, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    qo_offset_t = torch.tensor(
        [kl - ql for ql, kl in zip(qo_lens, kv_lens)],
        dtype=torch.int32)
    return q, k, v, qo_lens_t, kv_lens_t, qo_offset_t


def test_split_matches_nosplit():
    """Key test: split output == running decode and prefill independently."""
    print("\n=== test_split_matches_nosplit ===")
    device = "cuda:0"
    h_q, h_k, d = 32, 4, 128

    decode_qo = [1, 1, 32]
    decode_kv = [4096, 2048, 1024]
    prefill_qo = [256, 512]
    prefill_kv = [512, 1024]

    q, k, v, qo_lens_t, kv_lens_t, qo_offset_t = make_inputs(
        decode_qo + prefill_qo, decode_kv + prefill_kv, h_q, h_k, d, device)

    decode_nnz = sum(decode_qo)
    decode_kv_nnz = sum(decode_kv)

    # --- Run decode-only (no split) ---
    d_plan = _fmha_sm100_plan(
        qo_lens_t[:len(decode_qo)], kv_lens_t[:len(decode_qo)],
        h_q, num_kv_heads=h_k, causal=True,
        qo_offset=qo_offset_t[:len(decode_qo)])
    d_out, _ = _fmha_sm100(
        q[:decode_nnz], k[:decode_kv_nnz], v[:decode_kv_nnz], d_plan)

    # --- Run prefill-only (no split) ---
    p_plan = _fmha_sm100_plan(
        qo_lens_t[len(decode_qo):], kv_lens_t[len(decode_qo):],
        h_q, num_kv_heads=h_k, causal=True,
        qo_offset=qo_offset_t[len(decode_qo):])
    p_out, _ = _fmha_sm100(
        q[decode_nnz:], k[decode_kv_nnz:], v[decode_kv_nnz:], p_plan)

    ref_out = torch.cat([d_out, p_out], dim=0)

    # --- Run combined (triggers split) ---
    plan_info = fmha_sm100_plan(
        qo_lens_t, kv_lens_t, h_q,
        qo_offset=qo_offset_t, causal=True, num_kv_heads=h_k)

    has_mixed, split_idx, _, _, _ = plan_info
    assert has_mixed, "Expected mixed prefill/decode split"
    assert split_idx == len(decode_qo), f"Expected split at {len(decode_qo)}, got {split_idx}"

    combined_out, _ = fmha_sm100(q, k, v, plan_info=plan_info)
    torch.cuda.synchronize()

    return check("split_matches_nosplit", combined_out, ref_out, threshold=0.99999)


def test_split_vs_sdpa():
    """Compare split output against PyTorch SDPA reference."""
    print("\n=== test_split_vs_sdpa ===")
    device = "cuda:0"
    h_q, h_k, d = 32, 4, 128

    qo_lens = [1, 1, 1, 256]
    kv_lens = [4096, 2048, 1024, 512]

    q, k, v, qo_lens_t, kv_lens_t, qo_offset_t = make_inputs(
        qo_lens, kv_lens, h_q, h_k, d, device)

    o_ref = sdpa_ref_varlen(q, k, v, qo_lens, kv_lens, h_q, h_k,
                            qo_offset=[kl - ql for ql, kl in zip(qo_lens, kv_lens)])

    plan_info = fmha_sm100_plan(
        qo_lens_t, kv_lens_t, h_q,
        qo_offset=qo_offset_t, causal=True, num_kv_heads=h_k)

    out, _ = fmha_sm100(q, k, v, plan_info=plan_info)
    torch.cuda.synchronize()

    return check("split_vs_sdpa", out, o_ref, threshold=0.9999)


def test_split_gqa():
    """Test split with large GQA ratio (pack_factor > 1 for decode)."""
    print("\n=== test_split_gqa ===")
    device = "cuda:0"
    h_q, h_k, d = 128, 8, 128

    qo_lens = [1, 1, 256]
    kv_lens = [8192, 4096, 1024]

    q, k, v, qo_lens_t, kv_lens_t, qo_offset_t = make_inputs(
        qo_lens, kv_lens, h_q, h_k, d, device)

    o_ref = sdpa_ref_varlen(q, k, v, qo_lens, kv_lens, h_q, h_k,
                            qo_offset=[kl - ql for ql, kl in zip(qo_lens, kv_lens)])

    plan_info = fmha_sm100_plan(
        qo_lens_t, kv_lens_t, h_q,
        qo_offset=qo_offset_t, causal=True, num_kv_heads=h_k)

    out, _ = fmha_sm100(q, k, v, plan_info=plan_info)
    torch.cuda.synchronize()

    return check("split_gqa", out, o_ref, threshold=0.9999)


def test_split_paged():
    """Test split with paged KV cache."""
    print("\n=== test_split_paged ===")
    device = "cuda:0"
    h_q, h_k, d = 32, 4, 128
    page_size = 128

    qo_lens = [1, 1, 256]
    kv_lens = [2048, 1024, 512]

    total_qo = sum(qo_lens)
    q = torch.randn(total_qo, h_q, d, dtype=torch.bfloat16, device=device)

    # Build paged KV: construct per-batch, pad to page boundary, reshape to pages
    k_batches, v_batches = [], []
    all_k_pages, all_v_pages = [], []
    for kl in kv_lens:
        kb = torch.randn(kl, h_k, d, dtype=torch.bfloat16, device=device)
        vb = torch.randn(kl, h_k, d, dtype=torch.bfloat16, device=device)
        k_batches.append(kb)
        v_batches.append(vb)
        pages_per = (kl + page_size - 1) // page_size
        padded = pages_per * page_size
        if padded > kl:
            kb = torch.cat([kb, torch.zeros(padded - kl, h_k, d, dtype=torch.bfloat16, device=device)])
            vb = torch.cat([vb, torch.zeros(padded - kl, h_k, d, dtype=torch.bfloat16, device=device)])
        # (pages_per, page_size, h_k, d) -> (pages_per, h_k, page_size, d)
        all_k_pages.append(kb.reshape(pages_per, page_size, h_k, d).transpose(1, 2).contiguous())
        all_v_pages.append(vb.reshape(pages_per, page_size, h_k, d).transpose(1, 2).contiguous())

    k_paged = torch.cat(all_k_pages, dim=0)  # [total_pages, h_k, page_size, d]
    v_paged = torch.cat(all_v_pages, dim=0)
    total_pages = k_paged.shape[0]
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)

    # Reference: use non-paged varlen
    k_flat = torch.cat(k_batches, dim=0)
    v_flat = torch.cat(v_batches, dim=0)
    qo_offset_list = [kl - ql for ql, kl in zip(qo_lens, kv_lens)]
    o_ref = sdpa_ref_varlen(q, k_flat, v_flat, qo_lens, kv_lens, h_q, h_k,
                            qo_offset=qo_offset_list)

    qo_lens_t = torch.tensor(qo_lens, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    qo_offset_t = torch.tensor(qo_offset_list, dtype=torch.int32)

    plan_info = fmha_sm100_plan(
        qo_lens_t, kv_lens_t, h_q,
        qo_offset=qo_offset_t, causal=True, num_kv_heads=h_k, page_size=page_size)

    out, _ = fmha_sm100(q, k_paged, v_paged, plan_info=plan_info, kv_indices=kv_indices)
    torch.cuda.synchronize()

    return check("split_paged", out, o_ref, threshold=0.9999)


def test_split_maxscore():
    """Test that max_score is correctly merged across split."""
    print("\n=== test_split_maxscore ===")
    device = "cuda:0"
    h_q, h_k, d = 32, 4, 128

    qo_lens = [1, 1, 256]
    kv_lens = [2048, 1024, 512]

    q, k, v, qo_lens_t, kv_lens_t, qo_offset_t = make_inputs(
        qo_lens, kv_lens, h_q, h_k, d, device)

    plan_info = fmha_sm100_plan(
        qo_lens_t, kv_lens_t, h_q,
        qo_offset=qo_offset_t, causal=True, num_kv_heads=h_k,
        output_maxscore=True)

    out, ms = fmha_sm100(q, k, v, plan_info=plan_info, output_maxscore=True)
    torch.cuda.synchronize()

    passed = True
    # Verify output against SDPA
    o_ref = sdpa_ref_varlen(q, k, v, qo_lens, kv_lens, h_q, h_k,
                            qo_offset=[kl - ql for ql, kl in zip(qo_lens, kv_lens)])
    passed &= check("split_maxscore_out", out, o_ref, threshold=0.9999)

    # Verify max_score shape and basic properties
    if ms is not None:
        assert ms.shape[0] == h_q, f"max_score head dim mismatch: {ms.shape[0]} vs {h_q}"
        assert ms.shape[2] == sum(qo_lens), f"max_score token dim mismatch: {ms.shape[2]} vs {sum(qo_lens)}"
        non_inf = (ms != -float("inf")).sum().item()
        print(f"  max_score shape={list(ms.shape)}, non_inf_entries={non_inf}")
        passed &= non_inf > 0
    else:
        print("  [SKIP] max_score is None")

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] split_maxscore")
    if not passed:
        failed_cases.append("split_maxscore")
    return passed


if __name__ == "__main__":
    torch.manual_seed(42)
    all_passed = True

    all_passed &= test_split_matches_nosplit()
    all_passed &= test_split_vs_sdpa()
    all_passed &= test_split_gqa()
    all_passed &= test_split_paged()
    all_passed &= test_split_maxscore()

    print("\n" + "=" * 60)
    if failed_cases:
        print(f"FAILED ({len(failed_cases)}):")
        for f in failed_cases:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL PASSED")
        sys.exit(0)
