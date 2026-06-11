# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Test OnlyScore mode (SparseAttnMode::OnlyScore) pipeline correctness.

Calls the API with output_o=False, output_maxscore=True to trigger sparse_mode=2.
Compares max_score output against Full mode.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import math
import torch
from fmha_sm100 import fmha_sm100, fmha_sm100_plan

def _run(q, k, v, qo_lens, kv_lens, h_q, h_k, output_o=True, output_maxscore=True, causal=True):
    plan = fmha_sm100_plan(qo_lens, kv_lens, h_q, num_kv_heads=h_k,
                           causal=causal, output_maxscore=output_maxscore)
    torch.cuda.synchronize()
    o, ms = fmha_sm100(q, k, v, plan_info=plan,
                       output_o=output_o, output_maxscore=output_maxscore)
    torch.cuda.synchronize()
    return o, ms

def test_onlyscore_basic():
    device = "cuda:0"
    torch.manual_seed(42)

    h_q, h_k, d = 8, 8, 128
    qo_lens = [1, 4]
    kv_lens = [512, 1024]

    q = torch.randn(sum(qo_lens), h_q, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(sum(kv_lens), h_k, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(sum(kv_lens), h_k, d, device=device, dtype=torch.bfloat16)

    qo_t = torch.tensor(qo_lens, dtype=torch.int32)
    kv_t = torch.tensor(kv_lens, dtype=torch.int32)

    out, ms = _run(q, k, v, qo_t, kv_t, h_q, h_k, output_o=False, output_maxscore=True)

    print(f"  out is None: {out is None}")
    print(f"  max_score shape: {ms.shape if ms is not None else 'None'}")

    if ms is not None:
        non_inf = (ms != float('-inf')).sum().item()
        print(f"  max_score non-inf: {non_inf}/{ms.numel()}")
        finite = ms[ms != float('-inf')]
        if finite.numel() > 0:
            print(f"  max_score range: [{finite.min().item():.4f}, {finite.max().item():.4f}]")
        assert non_inf > 0, "All max_score values are -inf!"
    else:
        print("  [FAIL] max_score is None!")
        return False

    print("  [PASS] test_onlyscore_basic")
    return True

def test_onlyscore_sweep():
    device = "cuda:0"
    configs = [
        ([1], [128], 8, 8, 128, "single_1tok"),
        ([1], [256], 32, 4, 128, "gqa_1tok"),
        ([1, 1, 1, 1], [512, 256, 1024, 128], 8, 8, 128, "batch4_decode"),
        ([128], [2048], 8, 8, 128, "prefill_128"),
        ([256], [4096], 8, 8, 128, "prefill_256"),
        ([1, 256], [1024, 2048], 8, 8, 128, "mixed_decode_prefill"),
    ]

    all_pass = True
    for qo_lens, kv_lens, h_q, h_k, d, label in configs:
        torch.manual_seed(123)

        q = torch.randn(sum(qo_lens), h_q, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(sum(kv_lens), h_k, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(sum(kv_lens), h_k, d, device=device, dtype=torch.bfloat16)

        qo_t = torch.tensor(qo_lens, dtype=torch.int32)
        kv_t = torch.tensor(kv_lens, dtype=torch.int32)

        try:
            _, ms = _run(q, k, v, qo_t, kv_t, h_q, h_k, output_o=False, output_maxscore=True)

            if ms is None:
                print(f"  [{label}] FAIL - max_score is None")
                all_pass = False
                continue

            non_inf = (ms != float('-inf')).sum().item()
            has_nan = ms.isnan().any().item()

            if has_nan:
                print(f"  [{label}] FAIL - NaN in max_score!")
                all_pass = False
            elif non_inf == 0:
                print(f"  [{label}] FAIL - all -inf!")
                all_pass = False
            else:
                print(f"  [{label}] PASS  (non-inf={non_inf}/{ms.numel()})")

        except Exception as e:
            print(f"  [{label}] ERROR - {e}")
            all_pass = False

    return all_pass

def test_onlyscore_vs_full():
    device = "cuda:0"
    torch.manual_seed(42)

    h_q, h_k, d = 8, 8, 128
    qo_lens = [1, 4]
    kv_lens = [512, 1024]

    q = torch.randn(sum(qo_lens), h_q, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(sum(kv_lens), h_k, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(sum(kv_lens), h_k, d, device=device, dtype=torch.bfloat16)

    qo_t = torch.tensor(qo_lens, dtype=torch.int32)
    kv_t = torch.tensor(kv_lens, dtype=torch.int32)

    _, ms_full = _run(q, k, v, qo_t, kv_t, h_q, h_k, output_o=True, output_maxscore=True)
    _, ms_only = _run(q, k, v, qo_t, kv_t, h_q, h_k, output_o=False, output_maxscore=True)

    if ms_full is None or ms_only is None:
        print(f"  [FAIL] ms_full={ms_full is not None}, ms_only={ms_only is not None}")
        return False

    mask = (ms_full != float('-inf')) | (ms_only != float('-inf'))
    if mask.sum() == 0:
        print("  [SKIP] all -inf")
        return True

    diff = (ms_full[mask] - ms_only[mask]).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  Full vs OnlyScore max_score: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if max_diff > 0.01:
        print(f"  [FAIL] max_diff too large: {max_diff}")
        return False

    print("  [PASS] test_onlyscore_vs_full")
    return True


if __name__ == "__main__":
    print("=== OnlyScore Pipeline Tests ===")
    all_pass = True

    print("\n--- test_onlyscore_basic ---")
    all_pass &= test_onlyscore_basic()

    print("\n--- test_onlyscore_sweep ---")
    all_pass &= test_onlyscore_sweep()

    print("\n--- test_onlyscore_vs_full ---")
    all_pass &= test_onlyscore_vs_full()

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
