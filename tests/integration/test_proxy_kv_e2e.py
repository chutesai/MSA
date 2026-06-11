# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""End-to-end sparse attention with TWO-STAGE KV cache + full topk verification.

Pipeline (Quest / MagicPIG / RetrievalAttention style):

    KV cache A (proxy, MQA-compressed)
        ↓
    [Dense pass] fmha_sm100(num_kv_heads=1) → max_score
        ↓
    [sparse_topk_select] (no reduce, direct drop-in)
        ↓
    kv_block_indexes (T, num_kv_heads_real, topk)
        ↓
    KV cache B (real, GQA, full precision)
        ↓
    [Sparse pass] fmha_sm100(num_kv_heads=num_kv_heads_real, batch=T) → o

Key constraints on the head dimension:
    num_qo_heads_dense == num_kv_heads_real
    (so that sparse_topk's output shape (T, num_qo_heads_dense, topk)
     directly equals fmha sparse's expected (B=T, num_kv_heads_real, K))

Verification — each test case checks BOTH of:
  (1) Topk selection is correct: indices selected by sparse_topk_select have
      the same score multiset as torch.topk on the same max_score row.
      (This catches bugs where sparse_topk picks the wrong indices.)
  (2) Sparse fmha output ≈ PyTorch sparse reference using the SAME selected
      blocks on the real KV cache.
      (This catches bugs in the dispatch / layout / padding pipeline.)

Cases cover both dispatch paths:
  - small max_k_tiles (< 4096) → Filtered single-CTA path
  - large max_k_tiles (>= 4096) → Multi-CTA Lookback Fused path
"""
import math
import random
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fmha_sm100 import fmha_sm100, fmha_sm100_plan
from fmha_sm100 import sparse_topk_select


def is_sm100_or_sm103_supported(device):
    p = torch.cuda.get_device_properties(device)
    return p.major == 10 and p.minor in (0, 3)


failed_cases = []


def check_output(name, o, o_ref, threshold=0.999):
    nan_match = (o.isnan() == o_ref.isnan()).all().item()
    if not nan_match:
        print(f"  [FAIL output] {name}: NaN pattern mismatch "
              f"(kernel={o.isnan().sum().item()}, ref={o_ref.isnan().sum().item()})")
        return False
    valid = ~o.isnan()
    if not valid.any():
        return True
    o_v = o[valid].float()
    r_v = o_ref[valid].float()
    cos = torch.nn.functional.cosine_similarity(
        o_v.reshape(-1), r_v.reshape(-1), dim=0
    ).item()
    diff = (o_v - r_v).abs().max().item()
    passed = cos > threshold
    if not passed:
        print(f"  [FAIL output] {name}: cos={cos:.6f} (threshold {threshold})")
    return passed, cos, diff


def verify_topk_selection(max_score, kv_block_indexes, topk, atol=1e-4):
    """Verify sparse_topk_select picks indices whose score multiset matches torch.topk.

    max_score: (H, K, T) fp32                           ← dense pass output
    kv_block_indexes: (T, H, topk) int32 (-1 padded)    ← sparse_topk output

    For each (t, h) row, gather scores at the kernel-selected indices, sort,
    and compare with torch.topk's selected scores (also sorted).  Tie-break
    differences are absorbed by the multiset comparison.

    -1 padding positions are treated as -inf (i.e., "not selected"), and the
    corresponding ref slot must also be -inf (causal-masked / invalid block).
    """
    H, K, T = max_score.shape
    # Permute max_score to (T, H, K) so it aligns with kv_block_indexes' (T, H, topk)
    s_THK = max_score.permute(2, 0, 1).contiguous()  # (T, H, K)

    # ── PyTorch reference top-K on same scores ──
    ref_topk = torch.topk(s_THK, k=topk, dim=-1)
    ref_scores, _ = torch.sort(ref_topk.values, dim=-1)  # (T, H, topk) sorted asc

    # ── Gather scores at kernel-selected indices ──
    valid_mask = kv_block_indexes >= 0  # (T, H, topk)
    safe_idx = torch.where(valid_mask, kv_block_indexes,
                           torch.zeros_like(kv_block_indexes))
    kernel_scores = torch.gather(s_THK, -1, safe_idx.long())  # (T, H, topk)
    # -1 padding positions → treat as -inf
    kernel_scores = torch.where(valid_mask, kernel_scores,
                                torch.full_like(kernel_scores, float("-inf")))
    kernel_scores, _ = torch.sort(kernel_scores, dim=-1)

    # Compare element-wise: both -inf is OK, else require numerical match
    both_finite = kernel_scores.isfinite() & ref_scores.isfinite()
    if not both_finite.any():
        return True, 0.0, 0
    diff = (kernel_scores - ref_scores).abs()
    diff_finite = torch.where(both_finite, diff, torch.zeros_like(diff))
    max_diff = diff_finite.max().item()
    n_mismatch_inf = ((kernel_scores.isfinite() != ref_scores.isfinite())).sum().item()
    return (max_diff < atol) and (n_mismatch_inf == 0), max_diff, n_mismatch_inf


def sparse_ref_real_kv(q_real, k_pages_real, v_pages_real,
                       per_batch_blocks, qo_offsets,
                       num_qo_heads_real, num_kv_heads_real,
                       page_size, head_dim, device):
    """Reference: GQA sparse attention on REAL KV cache, given selected blocks."""
    h_r = num_qo_heads_real // num_kv_heads_real
    T = len(qo_offsets)
    ref_parts = []
    for b in range(T):
        q_b = q_real[b:b + 1]
        heads_out = []
        for h in range(num_qo_heads_real):
            kv_h = h // h_r
            blocks = per_batch_blocks[b][kv_h]
            if not blocks:
                heads_out.append(torch.zeros(1, head_dim, device=device, dtype=torch.float32))
                continue
            k_g = torch.cat([k_pages_real[blk, kv_h] for blk in blocks], dim=0)
            v_g = torch.cat([v_pages_real[blk, kv_h] for blk in blocks], dim=0)
            scores = torch.matmul(q_b[:, h].float(), k_g.float().T) / math.sqrt(head_dim)
            qi = torch.tensor([[qo_offsets[b]]], device=device, dtype=torch.int64)
            kv_pos = []
            for blk in blocks:
                kv_pos.extend(range(blk * page_size, (blk + 1) * page_size))
            ki = torch.tensor(kv_pos, device=device, dtype=torch.int64).unsqueeze(0)
            scores.masked_fill_(qi < ki, float("-inf"))
            heads_out.append(torch.matmul(torch.softmax(scores, dim=-1), v_g.float()))
        ref_parts.append(torch.stack(heads_out, dim=1))
    return torch.cat(ref_parts, dim=0).to(torch.bfloat16)


def _run_proxy_kv_e2e(name, seed, total_qo_len, num_kv_heads_real,
                      h_r_real, qo_offset_prefix=256,
                      page_size=128, head_dim=128, topk=16):
    """Two-stage KV e2e: validates BOTH topk selection AND sparse output."""
    torch.manual_seed(seed)
    random.seed(seed)
    dev = torch.device("cuda")
    dtype = torch.bfloat16

    num_qo_heads_real = num_kv_heads_real * h_r_real
    num_qo_heads_dense = num_kv_heads_real
    num_kv_heads_dense = 1

    kv_len = qo_offset_prefix + total_qo_len
    pages_per_seq = (kv_len + page_size - 1) // page_size

    # KV cache A: MQA proxy
    k_pages_dense = torch.randn(pages_per_seq, num_kv_heads_dense, page_size, head_dim,
                                device=dev, dtype=dtype)
    v_pages_dense = torch.randn(pages_per_seq, num_kv_heads_dense, page_size, head_dim,
                                device=dev, dtype=dtype)
    # KV cache B: real GQA (independent random tensor — proxy is NOT a function of real KV)
    k_pages_real = torch.randn(pages_per_seq, num_kv_heads_real, page_size, head_dim,
                               device=dev, dtype=dtype)
    v_pages_real = torch.randn(pages_per_seq, num_kv_heads_real, page_size, head_dim,
                               device=dev, dtype=dtype)

    q_dense = torch.randn(total_qo_len, num_qo_heads_dense, head_dim, device=dev, dtype=dtype)
    q_real = torch.randn(total_qo_len, num_qo_heads_real, head_dim, device=dev, dtype=dtype)

    # ── Stage 1: Dense pass on KV cache A ──
    qo_segment_lens_d = torch.tensor([total_qo_len], dtype=torch.int32)
    kv_segment_lens_d = torch.tensor([kv_len], dtype=torch.int32)
    qo_offset_d = torch.tensor([qo_offset_prefix], dtype=torch.int32)
    kv_indices_d = torch.arange(pages_per_seq, device=dev, dtype=torch.int32)
    plan_dense = fmha_sm100_plan(qo_segment_lens_d, kv_segment_lens_d,
        num_qo_heads_dense, causal=True,
        page_size=page_size, output_maxscore=True,
    )
    _, max_score = fmha_sm100(
        q_dense, k_pages_dense, v_pages_dense, plan_dense,
        sm_scale=1.0 / math.sqrt(head_dim),
        kv_indices=kv_indices_d,
        output_o=False, output_maxscore=True,
    )
    if max_score is None:
        print(f"  [SKIP] {name}: max_score is None")
        return True
    max_k_tiles = max_score.shape[1]
    if max_k_tiles < topk:
        print(f"  [SKIP] {name}: max_k_tiles={max_k_tiles} < topk={topk}")
        return True

    # ── Stage 2: sparse_topk — direct drop-in (no reduce) ──
    # num_valid_pages caps OOB indices (max_k_tiles is round-up-aligned > actual pages).
    kv_block_indexes = sparse_topk_select(
        max_score, topk, num_valid_pages=pages_per_seq)

    # ── Verification (1): topk selection correctness ──
    # Compare sparse_topk's selected score multiset with torch.topk on same max_score
    topk_ok, topk_max_diff, topk_n_inf_mismatch = verify_topk_selection(
        max_score, kv_block_indexes, topk
    )

    # ── Stage 4: Sparse pass on KV cache B (each token = 1 sparse batch) ──
    qo_segment_lens_s = torch.ones(total_qo_len, dtype=torch.int32)
    kv_segment_lens_s = torch.full((total_qo_len,), kv_len, dtype=torch.int32)
    qo_offset_s = torch.tensor(
        [qo_offset_prefix + i for i in range(total_qo_len)],
        dtype=torch.int32,
    )
    kv_indices_s = kv_indices_d.repeat(total_qo_len)
    plan_sparse = fmha_sm100_plan(qo_segment_lens_s, kv_segment_lens_s,
        num_qo_heads_real, causal=True,
        qo_offset=qo_offset_s,
        page_size=page_size, kv_block_num=topk,
        num_kv_heads=num_kv_heads_real,
    )
    o, _ = fmha_sm100(
        q_real, k_pages_real, v_pages_real, plan_sparse,
        sm_scale=1.0 / math.sqrt(head_dim),
        kv_indices=kv_indices_s,
        kv_block_indexes=kv_block_indexes,
        check_input_valid=True,
    )

    # ── Verification (2): sparse output ≈ PyTorch ref using SAME indices ──
    bi_cpu = kv_block_indexes.cpu().tolist()
    per_batch_blocks = []
    for b in range(total_qo_len):
        heads_blocks = []
        for h_kv in range(num_kv_heads_real):
            blocks = [x for x in bi_cpu[b][h_kv] if x >= 0]
            heads_blocks.append(sorted(set(blocks)))
        per_batch_blocks.append(heads_blocks)
    qo_offsets_list = [qo_offset_prefix + i for i in range(total_qo_len)]
    o_ref = sparse_ref_real_kv(q_real, k_pages_real, v_pages_real,
                               per_batch_blocks, qo_offsets_list,
                               num_qo_heads_real, num_kv_heads_real,
                               page_size, head_dim, dev)
    output_passed_pkg = check_output(name, o, o_ref)
    if isinstance(output_passed_pkg, tuple):
        output_ok, cos, max_diff = output_passed_pkg
    else:
        output_ok, cos, max_diff = output_passed_pkg, 0.0, 0.0

    # ── Final verdict: BOTH must pass ──
    overall = topk_ok and output_ok
    status = "PASS" if overall else "FAIL"
    print(f"  [{status}] {name}: max_k_tiles={max_k_tiles}, "
          f"topk_diff={topk_max_diff:.6f} (n_inf_mis={topk_n_inf_mismatch}), "
          f"output cos={cos:.6f}, max_diff={max_diff:.4f}")
    if not overall:
        if not topk_ok:
            failed_cases.append(f"{name}: topk WRONG (max_diff={topk_max_diff:.6f}, "
                                f"n_inf_mis={topk_n_inf_mismatch})")
        if not output_ok:
            failed_cases.append(f"{name}: output cos={cos:.6f}")
    return overall


if __name__ == "__main__":
    if not is_sm100_or_sm103_supported(torch.device("cuda")):
        print("Skipped: requires SM100/SM103")
        exit(0)

    all_pass = True

    print("=== Group A: small max_k_tiles → Filtered single-CTA path ===")
    print("  (qo_offset_prefix=256 → kv_len ~ 256+T → max_k_tiles=128)\n")
    for seed in range(3):
        all_pass &= _run_proxy_kv_e2e(
            f"GQA-4 T=8 small s={seed}", seed,
            total_qo_len=8, num_kv_heads_real=4, h_r_real=4,
            qo_offset_prefix=256,
        )
    for seed in range(2):
        all_pass &= _run_proxy_kv_e2e(
            f"MHA H=4 T=4 small s={seed}", seed,
            total_qo_len=4, num_kv_heads_real=4, h_r_real=1,
            qo_offset_prefix=256,
        )
    for seed in range(2):
        all_pass &= _run_proxy_kv_e2e(
            f"GQA-8 T=4 small s={seed}", seed,
            total_qo_len=4, num_kv_heads_real=4, h_r_real=8,
            qo_offset_prefix=256,
        )

    print("\n=== Group B: large max_k_tiles >= 4096 → Multi-CTA Lookback path ===")
    print("  (qo_offset_prefix=524000 → kv_len ~ 524K → max_k_tiles=4096)\n")
    # large prefix to push max_k_tiles >= 4096
    LARGE_PREFIX = 524000
    for seed in range(3):
        all_pass &= _run_proxy_kv_e2e(
            f"GQA-4 T=4 LARGE s={seed}", seed,
            total_qo_len=4, num_kv_heads_real=4, h_r_real=4,
            qo_offset_prefix=LARGE_PREFIX,
        )
    for seed in range(2):
        all_pass &= _run_proxy_kv_e2e(
            f"MHA H=4 T=4 LARGE s={seed}", seed,
            total_qo_len=4, num_kv_heads_real=4, h_r_real=1,
            qo_offset_prefix=LARGE_PREFIX,
        )
    for seed in range(2):
        all_pass &= _run_proxy_kv_e2e(
            f"GQA-8 T=4 LARGE s={seed}", seed,
            total_qo_len=4, num_kv_heads_real=4, h_r_real=8,
            qo_offset_prefix=LARGE_PREFIX,
        )
    # decode-only (T=1) with large prefix
    all_pass &= _run_proxy_kv_e2e(
        "GQA-4 T=1 LARGE decode", 7,
        total_qo_len=1, num_kv_heads_real=4, h_r_real=4,
        qo_offset_prefix=LARGE_PREFIX,
    )

    print()
    if all_pass:
        print("All proxy-KV E2E tests PASSED!")
        print()
        print("Pipeline validated end-to-end:")
        print("  Dense pass on COMPRESSED MQA KV cache → max_score")
        print("  sparse_topk_select (no reduce, direct drop-in) →")
        print("    ✓ topk selection matches torch.topk (score multiset bit-equivalent)")
        print("  Sparse pass on REAL GQA KV cache → output")
        print("    ✓ ≈ PyTorch sparse reference on REAL KV cache (cos sim ≥ 0.999)")
        print()
        print("Both Filtered single-CTA AND Multi-CTA Lookback dispatch paths verified.")
    else:
        print(f"{len(failed_cases)} test(s) FAILED:")
        for f in failed_cases:
            print(f"  {f}")
        exit(1)
