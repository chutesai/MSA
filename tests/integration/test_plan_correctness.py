#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Correctness tests for the plan kernel.

Validates:
  1. work_indptr monotonicity
  2. Tile index ranges
  3. Full tile coverage: every (batch, head, qo_tile) appears exactly once (nosplit)
     or has contiguous kv ranges that cover [0, total_kv_iters) (split)
  4. Split-index uniqueness: no duplicate sub_ids per (batch, head, qo_tile)
  5. No overlapping kv ranges within a tile

Usage:
    python tests/ops/sm100_fmha/test_plan_correctness.py
"""

import math
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fmha_sm100 import fmha_sm100_plan
from fmha_sm100.api import _compute_pack_factor


def check_plan(plan_info, qo_lens_list, kv_lens_list, num_qo_heads, num_kv_heads,
               qo_tile_size, kv_tile_size, causal, qo_offset_list=None, label=""):
    """Validate plan_info output."""
    errors = []

    wi = plan_info["work_indptr"].cpu()
    num_sms = wi.shape[0] - 1
    total_items = wi[-1].item()
    B = len(kv_lens_list)
    pack_factor = plan_info.get("pack_factor", 1)
    packed_hq = num_qo_heads // pack_factor if pack_factor > 1 else num_qo_heads

    # 1. work_indptr monotonically non-decreasing
    for i in range(num_sms):
        if wi[i + 1] < wi[i]:
            errors.append(f"work_indptr not monotonic: [{i}]={wi[i].item()} > [{i+1}]={wi[i+1].item()}")
            break

    if total_items == 0:
        errors.append("work_indptr last = 0 (no work items)")

    if total_items == 0 or errors:
        status = "FAIL"
        print(f"  [{status}] {label}")
        for e in errors[:10]:
            print(f"    ERROR: {e}")
        return False

    qt = plan_info["qo_tile_indices"][:total_items].cpu()
    hi = plan_info["head_indices"][:total_items].cpu()
    bi = plan_info["batch_indices"][:total_items].cpu()

    # 2. Index range checks
    for i in range(total_items):
        b_val = bi[i].item()
        if b_val < 0 or b_val >= B:
            errors.append(f"item {i}: batch={b_val} out of range [0, {B})")
        h_val = hi[i].item()
        if h_val < 0 or h_val >= packed_hq:
            errors.append(f"item {i}: head={h_val} out of range [0, {packed_hq})")
        if len(errors) > 10:
            break

    # 3-5. Coverage, uniqueness, overlap checks
    num_kv_splits = plan_info["num_kv_splits"]
    has_split = num_kv_splits > 1 and plan_info["kv_tile_begin_indices"] is not None

    if has_split:
        kb = plan_info["kv_tile_begin_indices"][:total_items].cpu()
        ke = plan_info["kv_tile_end_indices"][:total_items].cpu()
        si = plan_info["kv_split_indices"][:total_items].cpu()

        # Group by (batch, head, qo_tile)
        coverage = {}
        for i in range(total_items):
            key = (bi[i].item(), hi[i].item(), qt[i].item())
            coverage.setdefault(key, []).append((kb[i].item(), ke[i].item(), si[i].item()))

        for key, ranges in coverage.items():
            # Check kv_begin < kv_end
            for (rb, re, rs) in ranges:
                if rb >= re:
                    errors.append(f"tile {key}: kv_begin={rb} >= kv_end={re}")
                if rb < 0:
                    errors.append(f"tile {key}: kv_begin={rb} < 0")

            # Sort by kv_begin, check contiguous coverage from 0
            ranges.sort(key=lambda x: x[0])
            if ranges[0][0] != 0:
                errors.append(f"tile {key}: coverage starts at {ranges[0][0]}, not 0")
            for j in range(1, len(ranges)):
                if ranges[j][0] != ranges[j - 1][1]:
                    errors.append(f"tile {key}: gap/overlap [{ranges[j-1][1]}, {ranges[j][0]})")

            # Check sub_id uniqueness
            sub_ids = [r[2] for r in ranges]
            if len(sub_ids) != len(set(sub_ids)):
                errors.append(f"tile {key}: duplicate sub_ids {sub_ids}")

            if len(errors) > 20:
                break
    else:
        # Nosplit: each (batch, head, qo_tile) must appear exactly once
        seen = {}
        for i in range(total_items):
            key = (bi[i].item(), hi[i].item(), qt[i].item())
            if key in seen:
                errors.append(f"tile {key}: duplicate at items {seen[key]} and {i}")
            seen[key] = i

    # Check expected tile count
    expected_tiles = set()
    for b_idx in range(B):
        ql = qo_lens_list[b_idx]
        kl = kv_lens_list[b_idx]
        off_q = qo_offset_list[b_idx] if qo_offset_list else (kl - ql)
        for t in range(math.ceil(ql / qo_tile_size)):
            if causal and (t + 1) * qo_tile_size + off_q <= 0:
                continue
            eff_kv = min((t + 1) * qo_tile_size + off_q, kl) if causal else kl
            if eff_kv <= 0:
                continue
            for h in range(packed_hq):
                expected_tiles.add((b_idx, h, t))

    if not has_split:
        actual_tiles = set()
        for i in range(total_items):
            actual_tiles.add((bi[i].item(), hi[i].item(), qt[i].item()))
        missing = expected_tiles - actual_tiles
        extra = actual_tiles - expected_tiles
        if missing:
            errors.append(f"missing {len(missing)} tiles, e.g. {list(missing)[:3]}")
        if extra:
            errors.append(f"extra {len(extra)} tiles, e.g. {list(extra)[:3]}")
    else:
        actual_tiles = set(coverage.keys())
        missing = expected_tiles - actual_tiles
        extra = actual_tiles - expected_tiles
        if missing:
            errors.append(f"missing {len(missing)} tiles, e.g. {list(missing)[:3]}")
        if extra:
            errors.append(f"extra {len(extra)} tiles, e.g. {list(extra)[:3]}")

    status = "PASS" if not errors else "FAIL"
    items_per_sm = [(wi[i + 1] - wi[i]).item() for i in range(num_sms)]
    active_sms = sum(1 for x in items_per_sm if x > 0)
    print(f"  [{status}] {label}: {total_items} items, {active_sms} active SMs, "
          f"splits={num_kv_splits}, expected_tiles={len(expected_tiles)}")
    for e in errors[:10]:
        print(f"    ERROR: {e}")
    if len(errors) > 10:
        print(f"    ... and {len(errors) - 10} more errors")
    return len(errors) == 0


def run_test(B, Q, kv_len, H_Q, H_K, D, causal, page_size, num_kv_splits, label):
    device = "cuda:0"
    kv_lens_list = [kv_len] * B if isinstance(kv_len, int) else kv_len
    B = len(kv_lens_list)
    qo_lens_list = [Q] * B

    qo_lens = torch.full((B,), Q, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32)
    qo_offset = torch.tensor([kl - Q for kl in kv_lens_list], dtype=torch.int32)

    pf = _compute_pack_factor(Q, H_Q, H_K)
    qo_tile_size = 128 if Q * pf <= 128 else 256
    kv_tile_size = 256 if qo_tile_size == 128 else 128

    plan = fmha_sm100_plan(
        qo_lens, kv_lens, H_Q,
        causal=causal,
        qo_offset=qo_offset,
        page_size=page_size,
        num_kv_heads=H_K,
        num_kv_splits=num_kv_splits,
    )

    return check_plan(plan, qo_lens_list, kv_lens_list,
                      H_Q, H_K, qo_tile_size, kv_tile_size, causal,
                      label=label)


def main():
    torch.cuda.set_device(0)
    print("Plan kernel correctness tests")
    print("=" * 60)

    all_pass = True

    print("\n--- Nosplit tests ---")
    all_pass &= run_test(4, 4, 1024, 12, 2, 128, True, 128, 1,
                         "B=4 Q=4 kv=1024 nosplit causal")
    all_pass &= run_test(32, 4, 65536, 12, 2, 128, True, 128, 1,
                         "B=32 Q=4 kv=65536 nosplit causal")
    all_pass &= run_test(1, 1, 256, 12, 2, 128, True, 128, 1,
                         "B=1 Q=1 kv=256 nosplit causal")
    all_pass &= run_test(1, 4, 128, 12, 2, 128, False, 128, 1,
                         "B=1 Q=4 kv=128 nosplit non-causal")
    all_pass &= run_test(4, 4, [1024, 2048, 4096, 8192], 12, 2, 128, True, 128, 1,
                         "B=4 varied kv nosplit causal")

    print("\n--- Adaptive split tests ---")
    all_pass &= run_test(32, 4, 65536, 12, 2, 128, True, 128, -1,
                         "B=32 Q=4 kv=65536 adaptive causal")
    all_pass &= run_test(4, 4, [1024, 2048, 4096, 65536], 12, 2, 128, True, 128, -1,
                         "B=4 varied kv adaptive causal")
    all_pass &= run_test(1, 4, 65536, 48, 8, 128, True, 128, -1,
                         "B=1 Q=4 kv=65536 h48/8 adaptive")

    # Cases that previously failed
    print("\n--- Regression tests (previously failing cases) ---")
    all_pass &= run_test(1, 1, 131072, 48, 8, 128, True, 128, -1,
                         "B=1 Q=1 kv=131072 h48/8 adaptive")
    all_pass &= run_test(4, 6, 65536, 48, 8, 128, True, 128, -1,
                         "B=4 Q=6 kv=65536 h48/8 adaptive")
    all_pass &= run_test(32, 6, 512, 8, 8, 128, True, 128, -1,
                         "B=32 Q=6 kv=512 h8/8 adaptive")
    all_pass &= run_test(32, 6, 8192, 8, 8, 128, True, 128, -1,
                         "B=32 Q=6 kv=8192 h8/8 adaptive")

    # Edge cases: few tiles, many SMs
    print("\n--- Edge cases ---")
    all_pass &= run_test(1, 1, 256, 8, 8, 128, True, 128, -1,
                         "B=1 Q=1 kv=256 h8/8 adaptive (few tiles)")
    all_pass &= run_test(1, 1, 8192, 48, 8, 128, True, 128, -1,
                         "B=1 Q=1 kv=8192 h48/8 adaptive")
    all_pass &= run_test(128, 4, 65536, 48, 8, 128, True, 128, -1,
                         "B=128 Q=4 kv=65536 h48/8 adaptive (many tiles)")
    all_pass &= run_test(64, 4, 32768, 48, 8, 128, True, 128, -1,
                         "B=64 Q=4 kv=32768 h48/8 adaptive")

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
