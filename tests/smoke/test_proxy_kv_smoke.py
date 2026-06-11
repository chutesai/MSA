# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Minimal sparse attention E2E: max_score → sparse_topk_select → sparse fmha output.

Trimmed from test_sparse_attn_e2e_proxy_kv.py to the bare main path:

    KV cache A (proxy, MQA-compressed)
        ↓
    fmha_sm100 dense (num_kv_heads=1) → max_score
        ↓
    sparse_topk_select → kv_block_indexes (T, num_kv_heads_real, topk)
        ↓
    KV cache B (real, GQA, full precision)
        ↓
    fmha_sm100 sparse (batch=T, num_kv_heads=num_kv_heads_real) → o

Default: run the pipeline once + shape/dtype/NaN sanity checks.
With --check: also run the full topk-selection multiset check + PyTorch
              sparse-ref cosine check (ported from the e2e test).
"""
import argparse
import math
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select


def is_sm100_or_sm103_supported(device):
    p = torch.cuda.get_device_properties(device)
    return p.major == 10 and p.minor in (0, 3)


def verify_topk_selection(max_score, kv_block_indexes, topk, atol=1e-4):
    """Kernel-selected score multiset vs torch.topk on the same max_score."""
    s_THK = max_score.permute(2, 0, 1).contiguous()  # (T, H, K)
    ref_topk = torch.topk(s_THK, k=topk, dim=-1)
    ref_scores, _ = torch.sort(ref_topk.values, dim=-1)

    valid_mask = kv_block_indexes >= 0
    safe_idx = torch.where(valid_mask, kv_block_indexes, torch.zeros_like(kv_block_indexes))
    kernel_scores = torch.gather(s_THK, -1, safe_idx.long())
    kernel_scores = torch.where(valid_mask, kernel_scores,
                                torch.full_like(kernel_scores, float("-inf")))
    kernel_scores, _ = torch.sort(kernel_scores, dim=-1)

    both_finite = kernel_scores.isfinite() & ref_scores.isfinite()
    diff = (kernel_scores - ref_scores).abs()
    diff_finite = torch.where(both_finite, diff, torch.zeros_like(diff))
    max_diff = diff_finite.max().item()
    n_inf_mismatch = (kernel_scores.isfinite() != ref_scores.isfinite()).sum().item()
    return (max_diff < atol) and (n_inf_mismatch == 0), max_diff, n_inf_mismatch


def sparse_ref_real_kv(q_real, k_pages_real, v_pages_real,
                       per_batch_blocks, qo_offsets,
                       num_qo_heads_real, num_kv_heads_real,
                       page_size, head_dim, device):
    """GQA sparse attention PyTorch reference on REAL KV cache."""
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


def run(check=False, seed=0,
        total_qo_len=8, num_kv_heads_real=4, h_r_real=4,
        qo_offset_prefix=256, page_size=128, head_dim=128, topk=16):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    dtype = torch.bfloat16

    num_qo_heads_real = num_kv_heads_real * h_r_real
    num_qo_heads_dense = num_kv_heads_real
    num_kv_heads_dense = 1

    kv_len = qo_offset_prefix + total_qo_len
    pages_per_seq = (kv_len + page_size - 1) // page_size

    # KV cache A (MQA proxy) and KV cache B (real GQA, independent random)
    k_pages_dense = torch.randn(pages_per_seq, num_kv_heads_dense, page_size, head_dim,
                                device=dev, dtype=dtype)
    v_pages_dense = torch.randn(pages_per_seq, num_kv_heads_dense, page_size, head_dim,
                                device=dev, dtype=dtype)
    k_pages_real = torch.randn(pages_per_seq, num_kv_heads_real, page_size, head_dim,
                               device=dev, dtype=dtype)
    v_pages_real = torch.randn(pages_per_seq, num_kv_heads_real, page_size, head_dim,
                               device=dev, dtype=dtype)
    q_dense = torch.randn(total_qo_len, num_qo_heads_dense, head_dim, device=dev, dtype=dtype)
    q_real = torch.randn(total_qo_len, num_qo_heads_real, head_dim, device=dev, dtype=dtype)

    # ── Stage 1: Dense pass on KV cache A → max_score ──
    qo_segment_lens_d = torch.tensor([total_qo_len], dtype=torch.int32)
    kv_segment_lens_d = torch.tensor([kv_len], dtype=torch.int32)
    kv_indices_d = torch.arange(pages_per_seq, device=dev, dtype=torch.int32)
    plan_dense = fmha_sm100_plan(
        qo_segment_lens_d, kv_segment_lens_d, num_qo_heads_dense,
        causal=True, page_size=page_size,
        output_maxscore=True,
    )
    _, max_score = fmha_sm100(
        q_dense, k_pages_dense, v_pages_dense, plan_dense,
        sm_scale=1.0 / math.sqrt(head_dim), kv_indices=kv_indices_d,
        output_o=False, output_maxscore=True,
    )
    assert max_score is not None
    assert max_score.dim() == 3 and max_score.shape[0] == num_qo_heads_dense \
        and max_score.shape[2] == total_qo_len, f"max_score shape {tuple(max_score.shape)}"
    assert max_score.dtype == torch.float32 and not max_score.isnan().any()

    # ── Stage 2: sparse_topk_select → kv_block_indexes ──
    # num_valid_pages caps OOB indices (max_k_tiles is round-up-aligned > actual pages).
    kv_block_indexes = sparse_topk_select(
        max_score, topk, num_valid_pages=pages_per_seq)
    assert kv_block_indexes.shape == (total_qo_len, num_qo_heads_dense, topk)
    assert kv_block_indexes.dtype == torch.int32

    # ── Stage 3: Sparse pass on KV cache B → o (each token = 1 sparse batch) ──
    qo_segment_lens_s = torch.ones(total_qo_len, dtype=torch.int32)
    kv_segment_lens_s = torch.full((total_qo_len,), kv_len, dtype=torch.int32)
    qo_offset_s = torch.tensor(
        [qo_offset_prefix + i for i in range(total_qo_len)],
        dtype=torch.int32,
    )
    kv_indices_s = kv_indices_d.repeat(total_qo_len)
    plan_sparse = fmha_sm100_plan(
        qo_segment_lens_s, kv_segment_lens_s, num_qo_heads_real,
        causal=True, qo_offset=qo_offset_s,
        page_size=page_size, kv_block_num=topk,
        num_kv_heads=num_kv_heads_real,
    )
    o, _ = fmha_sm100(
        q_real, k_pages_real, v_pages_real, plan_sparse,
        sm_scale=1.0 / math.sqrt(head_dim), kv_indices=kv_indices_s,
        kv_block_indexes=kv_block_indexes, check_input_valid=True,
    )
    assert o.shape == (total_qo_len, num_qo_heads_real, head_dim)
    assert o.dtype == torch.bfloat16
    assert (~o.isnan()).any(), "all-NaN sparse output"

    print(f"[OK] max_score{tuple(max_score.shape)} (max_k_tiles={max_score.shape[1]}) "
          f"→ kv_block_indexes{tuple(kv_block_indexes.shape)} "
          f"→ o{tuple(o.shape)}")

    # ── Optional: full correctness check ──
    if check:
        topk_ok, topk_diff, n_inf_mis = verify_topk_selection(
            max_score, kv_block_indexes, topk)
        print(f"  topk select : max_diff={topk_diff:.6f}, n_inf_mis={n_inf_mis} "
              f"→ {'OK' if topk_ok else 'FAIL'}")

        bi_cpu = kv_block_indexes.cpu().tolist()
        per_batch_blocks = [
            [sorted({x for x in bi_cpu[b][h_kv] if x >= 0})
             for h_kv in range(num_kv_heads_real)]
            for b in range(total_qo_len)
        ]
        qo_offsets_list = [qo_offset_prefix + i for i in range(total_qo_len)]
        o_ref = sparse_ref_real_kv(
            q_real, k_pages_real, v_pages_real,
            per_batch_blocks, qo_offsets_list,
            num_qo_heads_real, num_kv_heads_real,
            page_size, head_dim, dev,
        )
        valid = ~o.isnan()
        cos = torch.nn.functional.cosine_similarity(
            o[valid].float().reshape(-1), o_ref[valid].float().reshape(-1), dim=0,
        ).item()
        out_ok = cos > 0.999
        print(f"  output cos  : {cos:.6f} (≥0.999) → {'OK' if out_ok else 'FAIL'}")
        assert topk_ok and out_ok


if __name__ == "__main__":
    if not is_sm100_or_sm103_supported(torch.device("cuda")):
        print("Skipped: requires SM100/SM103")
        raise SystemExit(0)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal sparse attention E2E "
                    "(max_score → sparse_topk_select → sparse fmha output).",
    )
    parser.add_argument("--check", action="store_true",
                        help="Also run topk-multiset + PyTorch sparse-ref cosine checks.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-qo-len", type=int, default=8,
                        help="T = number of query tokens. 1 = decode, large = prefill.")
    parser.add_argument("--num-kv-heads-real", type=int, default=4,
                        help="KV head count of the REAL GQA cache.")
    parser.add_argument("--h-r-real", type=int, default=4,
                        help="head replication: num_qo_heads_real = "
                             "num_kv_heads_real * h_r_real (1=MHA, 4=GQA-4, 8=GQA-8).")
    parser.add_argument("--qo-offset-prefix", type=int, default=256,
                        help="KV prefix length. 256 → max_k_tiles=128 (Filtered single-CTA "
                             "path); 524000 → max_k_tiles=4096 (Multi-CTA Lookback path).")
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    # NOTE: minfer.sparse_topk_select currently asserts topk == 16,
    # so we hard-fix it here rather than expose a misleading CLI flag.
    args = parser.parse_args()
    run(**vars(args))
