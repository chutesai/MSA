#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Correctness tests for q_offset_override in fmha_sm100.

Tests that overriding qo_offset at run time (without re-planning) produces
the same result as planning with that offset from the start.

Test scenarios:
  1. Override with int (uniform offset)
  2. Override with per-batch GPU tensor
  3. Plan with causal disabled (large offset), override with real offset at run time
  4. Override = None falls back to plan's offset
  5. Override in split_prefill_decode path (mixed short/long Q)
  6. Varlen (per-batch different q_len/kv_len) with per-batch override
"""

import sys
import random
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from fmha_sm100 import fmha_sm100, fmha_sm100_plan


def sdpa_ref(q_bf16, k_bf16, v_bf16, h_q, h_k, causal=True, qo_offset=None):
    """Reference attention via PyTorch SDPA, supports per-batch qo_offset."""
    b = q_bf16.shape[0]
    q_len, k_len = q_bf16.shape[1], k_bf16.shape[1]
    device = q_bf16.device
    gqa_ratio = h_q // h_k

    with torch.no_grad():
        need_custom_mask = causal and qo_offset is not None
        if need_custom_mask and isinstance(qo_offset, int) and qo_offset == 0:
            need_custom_mask = False

        mask = None
        if need_custom_mask:
            if isinstance(qo_offset, torch.Tensor):
                offset = qo_offset.to(device).reshape(b, 1, 1)
            else:
                offset = qo_offset
            row = torch.arange(q_len, device=device).reshape(1, q_len, 1) + offset
            col = torch.arange(k_len, device=device).reshape(1, 1, k_len)
            mask = (row >= col).unsqueeze(1)

        q_ref = q_bf16.transpose(1, 2)
        k_ref = k_bf16.transpose(1, 2).repeat_interleave(gqa_ratio, dim=1)
        v_ref = v_bf16.transpose(1, 2).repeat_interleave(gqa_ratio, dim=1)
        if mask is not None:
            o_ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, attn_mask=mask
            ).transpose(1, 2)
        else:
            o_ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, is_causal=causal
            ).transpose(1, 2)
    return o_ref.reshape(b * q_len, h_q, q_bf16.shape[-1])


def check(name, o, o_ref, threshold=0.99999):
    cos_sim = torch.nn.functional.cosine_similarity(
        o.float().reshape(-1), o_ref.float().reshape(-1), dim=0
    ).item()
    max_diff = (o.float() - o_ref.float()).abs().max().item()
    passed = cos_sim > threshold
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.4f}")
    return passed


failed_cases = []

def run_test(name, test_fn):
    global failed_cases
    try:
        passed = test_fn()
        if not passed:
            failed_cases.append(name)
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        failed_cases.append(name)


# ---------------------------------------------------------------------------
# Test 1: Override with int — same offset as plan, result must match
# ---------------------------------------------------------------------------
def test_override_int_same_as_plan():
    """Plan with offset=X, override with same int X → result identical to no-override."""
    print("\n=== Test 1: Override with int (same as plan) ===")
    device = "cuda:0"
    b, q_len, k_len, h_q, h_k, d = 4, 256, 2048, 32, 8, 128
    offset_val = k_len - q_len

    q = torch.randn(b * q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b * k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b * k_len, h_k, d, dtype=torch.bfloat16, device=device)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)
    qo_offset = torch.full((b,), offset_val, dtype=torch.int32)

    plan = fmha_sm100_plan(qo_lens, kv_lens, h_q, num_kv_heads=h_k,
                           qo_offset=qo_offset, device=device)

    o_baseline, _ = fmha_sm100(q, k, v, plan_info=plan)
    o_override_int, _ = fmha_sm100(q, k, v, plan_info=plan,
                                   q_offset_override=offset_val)
    torch.cuda.synchronize()
    return check("override_int == baseline", o_override_int, o_baseline)


# ---------------------------------------------------------------------------
# Test 2: Override with GPU tensor — same offset, result must match
# ---------------------------------------------------------------------------
def test_override_tensor_same_as_plan():
    """Plan with offset tensor, override with same tensor on GPU → identical."""
    print("\n=== Test 2: Override with GPU tensor (same as plan) ===")
    device = "cuda:0"
    b, q_len, k_len, h_q, h_k, d = 4, 128, 4096, 32, 8, 128
    offset_val = k_len - q_len

    q = torch.randn(b * q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(b * k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v = torch.randn(b * k_len, h_k, d, dtype=torch.bfloat16, device=device)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)
    qo_offset = torch.full((b,), offset_val, dtype=torch.int32)

    plan = fmha_sm100_plan(qo_lens, kv_lens, h_q, num_kv_heads=h_k,
                           qo_offset=qo_offset, device=device)

    o_baseline, _ = fmha_sm100(q, k, v, plan_info=plan)
    override_tensor = torch.full((b,), offset_val, dtype=torch.int32, device=device)
    o_override, _ = fmha_sm100(q, k, v, plan_info=plan,
                               q_offset_override=override_tensor)
    torch.cuda.synchronize()
    return check("override_tensor == baseline", o_override, o_baseline)


# ---------------------------------------------------------------------------
# Test 3: Plan with causal disabled (large offset), override with real offset
#          This is the key use case: plan once conservatively, override per-run.
# ---------------------------------------------------------------------------
def test_override_plan_no_causal_override_causal():
    """Plan with causal=False (no causal mask in plan), override with real offset.
    Result must match direct planning with that offset."""
    print("\n=== Test 3: Plan causal=False, override with real offset ===")
    device = "cuda:0"
    b, q_len, k_len, h_q, h_k, d = 2, 256, 2048, 32, 8, 128
    real_offset = k_len - q_len

    q_bf16 = torch.randn(b, q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    q = q_bf16.reshape(b * q_len, h_q, d)
    k = k_bf16.reshape(b * k_len, h_k, d)
    v = v_bf16.reshape(b * k_len, h_k, d)

    # Reference: SDPA with the real offset
    o_ref = sdpa_ref(q_bf16, k_bf16, v_bf16, h_q, h_k, causal=True, qo_offset=real_offset)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)

    # Plan WITHOUT causal — plan sees full KV range
    plan_wide = fmha_sm100_plan(qo_lens, kv_lens, h_q, num_kv_heads=h_k,
                                causal=False, device=device)

    # Override with real causal offset at run time
    override_tensor = torch.full((b,), real_offset, dtype=torch.int32, device=device)
    o_override, _ = fmha_sm100(q, k, v, plan_info=plan_wide,
                               q_offset_override=override_tensor)
    torch.cuda.synchronize()
    return check("plan_no_causal + override vs ref", o_override, o_ref, threshold=0.9999)


# ---------------------------------------------------------------------------
# Test 4: Override with a tighter (smaller) offset than plan
# ---------------------------------------------------------------------------
def test_override_tighter_offset():
    """Plan with offset=kv-qo (default), override with smaller offset.
    Smaller offset = tighter causal mask (sees fewer KV tokens)."""
    print("\n=== Test 4: Override with tighter offset ===")
    device = "cuda:0"
    b, q_len, k_len, h_q, h_k, d = 2, 128, 4096, 16, 4, 128
    plan_offset = k_len - q_len  # 3968 — widest causal
    real_offset = k_len - q_len - 512  # 3456 — tighter

    q_bf16 = torch.randn(b, q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    q = q_bf16.reshape(b * q_len, h_q, d)
    k = k_bf16.reshape(b * k_len, h_k, d)
    v = v_bf16.reshape(b * k_len, h_k, d)

    o_ref = sdpa_ref(q_bf16, k_bf16, v_bf16, h_q, h_k, causal=True, qo_offset=real_offset)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)
    plan_qo_offset = torch.full((b,), plan_offset, dtype=torch.int32)

    plan = fmha_sm100_plan(qo_lens, kv_lens, h_q, num_kv_heads=h_k,
                           qo_offset=plan_qo_offset, device=device)

    override_tensor = torch.full((b,), real_offset, dtype=torch.int32, device=device)
    o_override, _ = fmha_sm100(q, k, v, plan_info=plan,
                               q_offset_override=override_tensor)
    torch.cuda.synchronize()
    return check("tighter_offset vs ref", o_override, o_ref, threshold=0.9999)


# ---------------------------------------------------------------------------
# Test 5: Override = None — should use plan's offset (baseline)
# ---------------------------------------------------------------------------
def test_override_none():
    """q_offset_override=None should produce identical result to explicit plan offset."""
    print("\n=== Test 5: Override = None (use plan offset) ===")
    device = "cuda:0"
    b, q_len, k_len, h_q, h_k, d = 4, 512, 2048, 32, 8, 128
    offset_val = k_len - q_len

    q_bf16 = torch.randn(b, q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    q = q_bf16.reshape(b * q_len, h_q, d)
    k = k_bf16.reshape(b * k_len, h_k, d)
    v = v_bf16.reshape(b * k_len, h_k, d)

    o_ref = sdpa_ref(q_bf16, k_bf16, v_bf16, h_q, h_k, causal=True,
                     qo_offset=offset_val)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)
    qo_offset = torch.full((b,), offset_val, dtype=torch.int32)

    plan = fmha_sm100_plan(qo_lens, kv_lens, h_q, num_kv_heads=h_k,
                           qo_offset=qo_offset, device=device)

    o_none, _ = fmha_sm100(q, k, v, plan_info=plan, q_offset_override=None)
    torch.cuda.synchronize()
    return check("override=None vs ref", o_none, o_ref)


# ---------------------------------------------------------------------------
# Test 6: Override in split_prefill_decode path
# ---------------------------------------------------------------------------
def test_override_split_prefill_decode():
    """Mixed short-Q (decode) + long-Q (prefill) batches with override."""
    print("\n=== Test 6: Override in split_prefill_decode path ===")
    device = "cuda:0"
    h_q, h_k, d = 32, 8, 128

    # Batch 0,1: decode (q_len <= 128), Batch 2,3: prefill (q_len > 128)
    q_lens = [1, 64, 256, 512]
    kv_lens = [4096, 4096, 4096, 4096]
    b = len(q_lens)

    q_list, k_list, v_list, o_ref_list = [], [], [], []
    offsets = []
    for ql, kl in zip(q_lens, kv_lens):
        off = kl - ql
        offsets.append(off)
        qi = torch.randn(1, ql, h_q, d, dtype=torch.bfloat16, device=device)
        ki = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device)
        vi = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device)
        o_ref_i = sdpa_ref(qi, ki, vi, h_q, h_k, causal=True, qo_offset=off)
        q_list.append(qi.reshape(ql, h_q, d))
        k_list.append(ki.reshape(kl, h_k, d))
        v_list.append(vi.reshape(kl, h_k, d))
        o_ref_list.append(o_ref_i)

    q = torch.cat(q_list, dim=0)
    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)
    o_ref = torch.cat(o_ref_list, dim=0)

    qo_lens_t = torch.tensor(q_lens, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    qo_offset_t = torch.tensor(offsets, dtype=torch.int32)

    # Plan with the real offsets, split_prefill_decode=True
    plan = fmha_sm100_plan(qo_lens_t, kv_lens_t, h_q, num_kv_heads=h_k,
                           qo_offset=qo_offset_t, split_prefill_decode=True,
                           device=device)

    # Run with override = same offsets (on GPU)
    override_tensor = torch.tensor(offsets, dtype=torch.int32, device=device)
    o_override, _ = fmha_sm100(q, k, v, plan_info=plan,
                               q_offset_override=override_tensor)
    torch.cuda.synchronize()
    p1 = check("split_pd override_tensor vs ref", o_override, o_ref)

    # Run with override = int (uniform)
    # Use the largest offset so causal mask is within plan's mask for all batches
    max_offset = max(offsets)
    # Re-plan with uniform large offset so all batches have the same wide plan
    uniform_offset = torch.full((b,), max_offset, dtype=torch.int32)
    plan_wide = fmha_sm100_plan(qo_lens_t, kv_lens_t, h_q, num_kv_heads=h_k,
                                qo_offset=uniform_offset, split_prefill_decode=True,
                                device=device)
    o_override_int, _ = fmha_sm100(q, k, v, plan_info=plan_wide,
                                   q_offset_override=max_offset)
    # Compute ref with uniform offset
    o_ref_uniform_list = []
    for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
        qi = q_list[i].reshape(1, ql, h_q, d)
        ki = k_list[i].reshape(1, kl, h_k, d)
        vi = v_list[i].reshape(1, kl, h_k, d)
        o_ref_uniform_list.append(sdpa_ref(qi, ki, vi, h_q, h_k, causal=True, qo_offset=max_offset))
    o_ref_uniform = torch.cat(o_ref_uniform_list, dim=0)

    torch.cuda.synchronize()
    p2 = check("split_pd override_int vs ref", o_override_int, o_ref_uniform)
    return p1 and p2


# ---------------------------------------------------------------------------
# Test 7: Varlen with per-batch different offsets
# ---------------------------------------------------------------------------
def test_override_varlen_per_batch():
    """Varlen batches with different q_len/kv_len, plan with wide offset,
    override with per-batch tighter offsets."""
    print("\n=== Test 7: Varlen per-batch override ===")
    device = "cuda:0"
    h_q, h_k, d = 16, 4, 128
    random.seed(123)
    torch.manual_seed(123)

    b = 6
    q_lens, kv_lens, real_offsets = [], [], []
    for _ in range(b):
        ql = random.choice([1, 32, 64, 128])
        kl = random.randint(ql, min(8192, ql * 64))
        off = random.randint(0, kl - ql)
        q_lens.append(ql)
        kv_lens.append(kl)
        real_offsets.append(off)

    # Sort by q_len (required for split_prefill_decode)
    order = sorted(range(b), key=lambda i: q_lens[i])
    q_lens = [q_lens[i] for i in order]
    kv_lens = [kv_lens[i] for i in order]
    real_offsets = [real_offsets[i] for i in order]

    q_list, k_list, v_list, o_ref_list = [], [], [], []
    for ql, kl, off in zip(q_lens, kv_lens, real_offsets):
        qi = torch.randn(1, ql, h_q, d, dtype=torch.bfloat16, device=device)
        ki = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device)
        vi = torch.randn(1, kl, h_k, d, dtype=torch.bfloat16, device=device)
        o_ref_i = sdpa_ref(qi, ki, vi, h_q, h_k, causal=True, qo_offset=off)
        q_list.append(qi.reshape(ql, h_q, d))
        k_list.append(ki.reshape(kl, h_k, d))
        v_list.append(vi.reshape(kl, h_k, d))
        o_ref_list.append(o_ref_i)

    q = torch.cat(q_list, dim=0)
    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)
    o_ref = torch.cat(o_ref_list, dim=0)

    qo_lens_t = torch.tensor(q_lens, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    # Plan with wide offsets (kv_len - qo_len per batch, the maximum possible)
    wide_offsets = [kl - ql for ql, kl in zip(q_lens, kv_lens)]
    plan_offset_t = torch.tensor(wide_offsets, dtype=torch.int32)
    plan = fmha_sm100_plan(qo_lens_t, kv_lens_t, h_q, num_kv_heads=h_k,
                           qo_offset=plan_offset_t, device=device)

    # Override with tighter real offsets
    override_t = torch.tensor(real_offsets, dtype=torch.int32, device=device)
    o_override, _ = fmha_sm100(q, k, v, plan_info=plan,
                               q_offset_override=override_t)
    torch.cuda.synchronize()
    return check("varlen per-batch override vs ref", o_override, o_ref, threshold=0.9999)


# ---------------------------------------------------------------------------
# Test 8: Decode (q_len=1) with override
# ---------------------------------------------------------------------------
def test_override_decode():
    """Decode scenario: q_len=1, override offset."""
    print("\n=== Test 8: Decode (q_len=1) with override ===")
    device = "cuda:0"
    b, q_len, k_len, h_q, h_k, d = 8, 1, 4096, 32, 8, 128
    real_offset = k_len - q_len

    q_bf16 = torch.randn(b, q_len, h_q, d, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    v_bf16 = torch.randn(b, k_len, h_k, d, dtype=torch.bfloat16, device=device)
    q = q_bf16.reshape(b * q_len, h_q, d)
    k = k_bf16.reshape(b * k_len, h_k, d)
    v = v_bf16.reshape(b * k_len, h_k, d)

    o_ref = sdpa_ref(q_bf16, k_bf16, v_bf16, h_q, h_k, causal=True,
                     qo_offset=real_offset)

    qo_lens = torch.full((b,), q_len, dtype=torch.int32)
    kv_lens = torch.full((b,), k_len, dtype=torch.int32)
    qo_offset = torch.full((b,), real_offset, dtype=torch.int32)

    plan = fmha_sm100_plan(qo_lens, kv_lens, h_q, num_kv_heads=h_k,
                           qo_offset=qo_offset, device=device)

    # Override with int
    o_int, _ = fmha_sm100(q, k, v, plan_info=plan,
                          q_offset_override=real_offset)
    torch.cuda.synchronize()
    p1 = check("decode override_int vs ref", o_int, o_ref)

    # Override with GPU tensor
    override_t = torch.full((b,), real_offset, dtype=torch.int32, device=device)
    o_tensor, _ = fmha_sm100(q, k, v, plan_info=plan,
                             q_offset_override=override_t)
    torch.cuda.synchronize()
    p2 = check("decode override_tensor vs ref", o_tensor, o_ref)
    return p1 and p2


def main():
    print("=" * 60)
    print("  q_offset_override correctness tests")
    print("=" * 60)

    tests = [
        ("1: override int same as plan",           test_override_int_same_as_plan),
        ("2: override tensor same as plan",         test_override_tensor_same_as_plan),
        ("3: plan no-causal, override with offset", test_override_plan_no_causal_override_causal),
        ("4: tighter offset override",              test_override_tighter_offset),
        ("5: override=None baseline",               test_override_none),
        ("6: split_prefill_decode override",         test_override_split_prefill_decode),
        ("7: varlen per-batch override",             test_override_varlen_per_batch),
        ("8: decode q_len=1 override",               test_override_decode),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print("\n" + "=" * 60)
    if failed_cases:
        print(f"  FAILED ({len(failed_cases)}/{len(tests)}):")
        for f in failed_cases:
            print(f"    - {f}")
    else:
        print(f"  ALL {len(tests)} TESTS PASSED")
    print("=" * 60)

    return len(failed_cases) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
