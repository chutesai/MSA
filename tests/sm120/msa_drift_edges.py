#!/usr/bin/env python3
"""Adversarial drift-regime edge suite for the new MSA qstat/CSR path.

Context: production C=8 jump-drift-merge sync NaNs the new MSA path at 4-9k
steps while DDP is clean for 78k. This suite feeds the kernels the degenerate
inputs that drift can produce and smooth training never does:

  E1  empty / thin selections (all -1 rows, single-block rows)
  E2  fp8 scale edges (amax=0 blocks, zero/inf/nan injected scales,
      fresh-vs-STALE scale simulation of a post-merge activation spike)
  E3  post-merge logit spikes (q x10 / x100)
  E4  CSR invariant fuzz (max-skew, boundary, empty-kv patterns)

Every case reports isfinite() of out/dq/dk/dv (+ lse where available) and
never asserts — it prints a PASS/EDGE table so the whole space is mapped in
one run. Run from ~/MSA/tests/sm120 with venv-msa + MSA env exports.
"""
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_qstat import _build  # noqa: E402
from test_triton_forward import _make_q2k  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "python", "fmha_sm100", "cute")))
from interface import sparse_atten_func  # noqa: E402
from sparse_index_utils import build_k2q_csr  # noqa: E402
from src.sm120.qstat import (  # noqa: E402
    quantize_kv_fp8_scaled,
    sparse_attention_qstat,
    sparse_attention_qstat_fp8,
)

RESULTS = []


def finite(name, *tensors):
    bad = []
    for i, t in enumerate(tensors):
        if t is None:
            continue
        tf = t.float()
        n_nan = torch.isnan(tf).sum().item()
        n_inf = torch.isinf(tf).sum().item()
        if n_nan or n_inf:
            bad.append(f"t{i}: nan={n_nan} inf={n_inf}")
    return bad


def record(case, status, detail=""):
    RESULTS.append((case, status, detail))
    print(f"[{status:5s}] {case}  {detail}", flush=True)


def run_case(case, fn):
    try:
        bad = fn()
        if bad:
            record(case, "EDGE", "; ".join(bad))
        else:
            record(case, "PASS")
    except Exception as exc:
        record(case, "CRASH", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def _csr(q2k, cu, seq, batch, g, blk_kv=128):
    total = batch * seq
    return build_k2q_csr(
        q2k, cu, cu, blk_kv, total_k=total, max_seqlen_k=seq,
        max_seqlen_q=seq, total_rows=batch * ((seq + blk_kv - 1) // blk_kv),
        qhead_per_kv=g, return_schedule=True,
    )


def _fwd_bwd(impl, q, k, v, dout, cu, q2k, row, idx, schedule, topk, seq,
             want_lse=False):
    # impl: "triton" | "cuda" via FMHA_SM120_QSTAT_IMPL (the dev's knob).
    old = os.environ.get("FMHA_SM120_QSTAT_IMPL")
    os.environ["FMHA_SM120_QSTAT_IMPL"] = impl
    try:
        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        if want_lse:
            out, lse = sparse_atten_func(
                qr, kr, vr, row, idx, topk, blk_kv=128, causal=True,
                cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=seq,
                max_seqlen_k=seq, schedule=schedule, q2k_indices=q2k,
                return_softmax_lse=True)
        else:
            out = sparse_atten_func(
                qr, kr, vr, row, idx, topk, blk_kv=128, causal=True,
                cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=seq,
                max_seqlen_k=seq, schedule=schedule, q2k_indices=q2k)
            lse = None
        if out.grad_fn is None:
            record_note = "NO-GRAPH (fwd returned detached tensor)"
            torch.cuda.synchronize()
            return out.detach(), lse, None, None, None, record_note
        out.backward(dout)
        torch.cuda.synchronize()
        return out.detach(), lse, qr.grad, kr.grad, vr.grad, ""
    finally:
        if old is None:
            os.environ.pop("FMHA_SM120_QSTAT_IMPL", None)
        else:
            os.environ["FMHA_SM120_QSTAT_IMPL"] = old


# ============================= E1: selections =============================
def e1():
    batch, seq, head_kv, g, topk = 1, 512, 2, 4, 4
    base = _build(batch, seq, head_kv, g, topk, seed=101)
    q, k, v, dout, cu, q2k, _, _, _ = base

    def lse_probe(lse):
        if lse is None:
            return []
        zl = (lse.float() ** 2).mean()
        if not torch.isfinite(zl).item():
            return [f"LSE-CONSUMER poisoned: mean(lse^2)={zl.item()}"]
        return []

    def variant(tag, q2k_mod):
        row, idx, schedule = _csr(q2k_mod, cu, seq, batch, g)
        # arm 1: torch-reference path (lse + autograd)
        def go_ref():
            out, lse, dq, dk, dv, note = _fwd_bwd(
                "triton", q, k, v, dout, cu, q2k_mod, row, idx, schedule,
                topk, seq, want_lse=True)
            bad = finite("x", out, lse, dq, dk, dv) + lse_probe(lse)
            if note:
                bad.append(note)
            return bad
        run_case(f"E1/{tag}/ref-lse", go_ref)
        # arm 2: the REAL training path (triton autograd, no lse return)
        def go_train():
            out, lse, dq, dk, dv, note = _fwd_bwd(
                "triton", q, k, v, dout, cu, q2k_mod, row, idx, schedule,
                topk, seq, want_lse=False)
            bad = finite("x", out, dq, dk, dv)
            if note:
                bad.append(note)
            return bad
        run_case(f"E1/{tag}/train", go_train)
        # arm 3: the qstat production entry (both impls)
        for impl in ("triton", "cuda"):
            def go_qstat(impl=impl):
                old_i = os.environ.get("FMHA_SM120_QSTAT_IMPL")
                os.environ["FMHA_SM120_QSTAT_IMPL"] = impl
                try:
                    qr = q.detach().clone().requires_grad_(True)
                    kr = k.detach().clone().requires_grad_(True)
                    vr = v.detach().clone().requires_grad_(True)
                    out, lse = sparse_attention_qstat(
                        qr, kr, vr, q2k_mod, row, idx, topk=topk, blk_kv=128,
                        cu_seqlens_q=cu, cu_seqlens_k=cu,
                        return_softmax_lse=True)
                    out.backward(dout)
                    torch.cuda.synchronize()
                    return (finite("x", out, lse, qr.grad, kr.grad, vr.grad)
                            + lse_probe(lse))
                finally:
                    if old_i is None:
                        os.environ.pop("FMHA_SM120_QSTAT_IMPL", None)
                    else:
                        os.environ["FMHA_SM120_QSTAT_IMPL"] = old_i
            run_case(f"E1/{tag}/qstat-{impl}", go_qstat)

    # (a) one full query block selects NOTHING in every head
    q2k_a = q2k.clone()
    q2k_a[:, 128:256, :] = -1
    variant("empty-qblock", q2k_a)
    # (b) sprinkle empty rows at boundaries (first/last tokens)
    q2k_b = q2k.clone()
    q2k_b[:, :2, :] = -1
    q2k_b[:, -2:, :] = -1
    variant("empty-edges", q2k_b)
    # (c) thin: every row keeps exactly one block
    q2k_c = q2k.clone()
    q2k_c[:, :, 1:] = -1
    variant("single-block", q2k_c)
    # (d) fully empty selection everywhere (extreme)
    q2k_d = torch.full_like(q2k, -1)
    variant("all-empty", q2k_d)


# ============================= E2: fp8 edges ==============================
def e2():
    batch, seq, head_kv, g, topk = 1, 512, 2, 4, 4
    q, k, v, dout, cu, q2k, row, idx, schedule = _build(
        batch, seq, head_kv, g, topk, seed=102)

    def run_fp8(tag, k_in, v_in, mutate=None):
        def go():
            k_u8, v_u8, ks, vs = quantize_kv_fp8_scaled(k_in, v_in)
            if mutate is not None:
                mutate(k_u8, v_u8, ks, vs)
            sbad = finite("scales", ks, vs)
            k_deq = (k_u8.view(torch.float8_e4m3fn).float()
                     * ks.unsqueeze(-1)).to(torch.bfloat16)
            v_deq = (v_u8.view(torch.float8_e4m3fn).float()
                     * vs.unsqueeze(0)).to(torch.bfloat16)
            qr = q.detach().clone().requires_grad_(True)
            kr = k_deq.detach().clone().requires_grad_(True)
            vr = v_deq.detach().clone().requires_grad_(True)
            out = sparse_attention_qstat_fp8(
                qr, kr, vr, k_u8, v_u8, ks, vs, q2k, row, idx, topk=topk,
                blk_kv=128, cu_seqlens_q=cu, cu_seqlens_k=cu)
            out.backward(dout)
            torch.cuda.synchronize()
            return (["QUANT:" + b for b in sbad]
                    + finite("x", out, qr.grad, kr.grad, vr.grad))
        run_case(f"E2/{tag}", go)

    run_fp8("baseline", k, v)
    # (a) all-zero K region -> amax=0 path in the quantizer
    kz = k.clone(); kz[100:180] = 0.0
    run_fp8("amax0-kblock", kz, v)
    kz2 = k.clone(); kz2[:, 1, :] = 0.0        # a whole head all-zero
    run_fp8("amax0-khead", kz2, v)
    vz = v.clone(); vz[:] = 0.0                # extreme: v identically 0
    run_fp8("amax0-v-all", k, vz)
    # (b) spiked K with FRESH scales (construction-safety check)
    ksp = k.clone(); ksp[300:310] *= 1000.0
    run_fp8("spike-fresh-scale", ksp, v)
    # (c) STALE scale: quantize the calm tensor, then pretend the data was
    #     spiked — emulate by inflating the stored u8 dequant target via a
    #     scale that is too small for the true data (the drift mechanism).
    def stale(k_u8, v_u8, ks, vs):
        ks *= 0.001                            # scales stale-small by 1000x
    run_fp8("stale-scale-sim", ksp, v, mutate=stale)
    # (d) direct metadata poison: zero / inf / nan single scale entries
    def s_zero(k_u8, v_u8, ks, vs): ks[5] = 0.0
    def s_inf(k_u8, v_u8, ks, vs): ks[6] = float("inf")
    def s_nan(k_u8, v_u8, ks, vs): ks[7] = float("nan")
    run_fp8("scale-zero", k, v, mutate=s_zero)
    run_fp8("scale-inf", k, v, mutate=s_inf)
    run_fp8("scale-nan", k, v, mutate=s_nan)


# ============================= E3: spikes =================================
def e3():
    batch, seq, head_kv, g, topk = 1, 512, 2, 4, 4
    q, k, v, dout, cu, q2k, row, idx, schedule = _build(
        batch, seq, head_kv, g, topk, seed=103)
    for mult in (10.0, 100.0):
        qs = (q.float() * mult).to(torch.bfloat16)
        for backend in ("triton", "cuda"):
            def go():
                out, lse, dq, dk, dv, note = _fwd_bwd(
                    backend, qs, k, v, dout, cu, q2k, row, idx, schedule,
                    topk, seq, want_lse=True)
                bad = finite("x", out, lse, dq, dk, dv)
                if note:
                    bad.append(note)
                return bad
            run_case(f"E3/qx{int(mult)}/{backend}", go)


# ============================= E4: CSR fuzz ===============================
def csr_invariants(q2k, row, idx, total_q, g, blk_kv, seq, batch):
    # row is a PER-HEAD rowptr [head_kv, rows+1]; idx counts one entry per
    # valid (kv-head, q-token, block) mapping.
    bad = []
    rowc = row.detach().cpu()
    if rowc.ndim == 1:
        rowc = rowc.unsqueeze(0)
    idxc = idx.detach().cpu().reshape(-1)
    for h in range(rowc.shape[0]):
        r = rowc[h]
        if not torch.all(r[1:] >= r[:-1]):
            bad.append(f"rowptr head {h} not monotone")
    valid_idx = int((idxc >= 0).sum().item())
    want = int((q2k >= 0).sum().item())
    if valid_idx != want:
        bad.append(f"count mismatch: valid idx entries={valid_idx} "
                   f"!= valid q2k entries={want}")
    return bad


def e4():
    batch, seq, head_kv, g, topk = 1, 512, 2, 4, 4
    q, k, v, dout, cu, q2k0, _, _, _ = _build(batch, seq, head_kv, g, topk,
                                              seed=104)
    total = batch * seq
    nblk = (seq + 127) // 128

    def variant(tag, q2k_mod):
        def go():
            row, idx, schedule = _csr(q2k_mod, cu, seq, batch, g)
            bad = csr_invariants(q2k_mod, row, idx, total, g, 128, seq, batch)
            out, lse, dq, dk, dv, note = _fwd_bwd(
                "triton", q, k, v, dout, cu, q2k_mod, row, idx, schedule,
                topk, seq, want_lse=False)
            if note:
                bad.append(note)
            return bad + finite("x", out, dq, dk, dv)
        run_case(f"E4/{tag}", go)

    # (a) max skew: every query selects block 0 only
    qa = torch.zeros_like(q2k0); qa[:, :, 1:] = -1
    variant("all-to-block0", qa)
    # (b) all queries select the LAST (partial-boundary) block only
    qb = torch.full_like(q2k0, -1); qb[:, :, 0] = nblk - 1
    variant("all-to-lastblock", qb)
    # (c) alternating: even tokens -> block0, odd -> last block
    qc = torch.full_like(q2k0, -1)
    qc[:, 0::2, 0] = 0
    qc[:, 1::2, 0] = nblk - 1
    variant("bimodal-skew", qc)
    # (d) duplicate block ids within a row (contract violation probe)
    qd = q2k0.clone(); qd[:, :, 1] = qd[:, :, 0]
    variant("dup-in-row", qd)


if __name__ == "__main__":
    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(0), flush=True)
    e1()
    e2()
    e3()
    e4()
    print("\n===== SUMMARY =====")
    n_edge = 0
    for case, status, detail in RESULTS:
        if status != "PASS":
            n_edge += 1
            print(f"  {status:5s} {case}: {detail}")
    print(f"{len(RESULTS)} cases, {n_edge} non-PASS")
    print("DRIFT_EDGES_DONE")
