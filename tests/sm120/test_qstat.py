#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Correctness of the Q-stationary (qstat) SM120 training backend.

Small-sequence cases are checked against the torch reference. The large
dense-coverage case is instead gated on agreement with the CSR Triton
backward: both are BF16 kernels sharing the same accumulation-noise floor,
and at 1000-term sums that floor exceeds any tolerance tight enough to catch
real indexing bugs against an fp32 oracle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from interface import sparse_atten_func  # noqa: E402
from sparse_index_utils import build_k2q_csr  # noqa: E402
from src.sm120.qstat import (  # noqa: E402
    quantize_kv_fp8_scaled,
    sparse_attention_qstat,
    sparse_attention_qstat_fp8,
)
from test_triton_forward import _make_q2k  # noqa: E402


def _build(batch, seq, head_kv, g, topk, seed, blk_kv=128, dim=128):
    torch.manual_seed(seed)
    dev = "cuda"
    head_q = head_kv * g
    total = batch * seq
    q = torch.randn(total, head_q, dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(total, head_kv, dim, device=dev, dtype=torch.bfloat16)
    v = torch.randn(total, head_kv, dim, device=dev, dtype=torch.bfloat16)
    dout = torch.randn_like(q)
    cu = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * seq
    q2k = _make_q2k(head_kv, (seq,) * batch, topk, blk_kv=blk_kv, device=dev)
    row, idx, schedule = build_k2q_csr(
        q2k, cu, cu, blk_kv, total_k=total, max_seqlen_k=seq, max_seqlen_q=seq,
        total_rows=batch * ((seq + blk_kv - 1) // blk_kv), qhead_per_kv=g,
        return_schedule=True,
    )
    return q, k, v, dout, cu, q2k, row, idx, schedule


def _grads_backend(backend_env, q, k, v, dout, cu, q2k, row, idx, schedule, topk, seq):
    old = {k_: os.environ.get(k_) for k_ in ("FMHA_SM120_BACKEND", "FMHA_SM120_TRITON_MODE")}
    try:
        os.environ["FMHA_SM120_BACKEND"] = backend_env
        os.environ.pop("FMHA_SM120_TRITON_MODE", None)
        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        out = sparse_atten_func(
            qr, kr, vr, row, idx, topk, blk_kv=128, causal=True,
            cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=seq, max_seqlen_k=seq,
            schedule=schedule, q2k_indices=q2k,
        )
        out.backward(dout)
    finally:
        for k_, v_ in old.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
    return out.detach(), qr.grad, kr.grad, vr.grad


def _grads_qstat(q, k, v, dout, cu, q2k, row, idx, topk):
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    out = sparse_attention_qstat(
        qr, kr, vr, q2k, row, idx, topk=topk, blk_kv=128,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
    )
    out.backward(dout)
    return out.detach(), qr.grad, kr.grad, vr.grad


def run_case(*, batch, seq, head_kv, g, topk, seed):
    args = _build(batch, seq, head_kv, g, topk, seed)
    q, k, v, dout, cu, q2k, row, idx, schedule = args
    r_out, r_dq, r_dk, r_dv = _grads_backend(
        "torch_ref", q, k, v, dout, cu, q2k, row, idx, schedule, topk, seq
    )
    _, _, c_dk, c_dv = _grads_backend(
        "triton", q, k, v, dout, cu, q2k, row, idx, schedule, topk, seq
    )
    out, dq, dk, dv = _grads_qstat(q, k, v, dout, cu, q2k, row, idx, topk)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), r_out.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dq.float(), r_dq.float(), rtol=8e-2, atol=5e-2)
    # dK/dV sums share the BF16 accumulation-noise floor with the CSR kernels
    # (which grows with the GQA group since more query heads accumulate into
    # each element): require tight agreement with the CSR kernels plus a
    # mean-deviation bound against the fp32 oracle, which stays flat under
    # noise but explodes on any systematic indexing or scaling bug.
    for name, got, csr, ref in (("dk", dk, c_dk, r_dk), ("dv", dv, c_dv, r_dv)):
        d = (got.float() - csr.float()).abs().max().item()
        assert d < 0.1, f"{name} deviates from csr backward: {d}"
        m = (got.float() - ref.float()).abs().mean().item()
        assert m < 0.02, f"{name} mean deviation vs fp32 reference: {m}"
    print(f"ok qstat batch={batch} seq={seq} g={g} topk={topk}")


def test_qstat_matches_reference() -> None:
    run_case(batch=2, seq=512, head_kv=2, g=4, topk=4, seed=31)
    run_case(batch=1, seq=512, head_kv=2, g=8, topk=4, seed=32)
    run_case(batch=1, seq=512, head_kv=1, g=16, topk=4, seed=33)


def test_qstat_agrees_with_csr_backward_at_scale() -> None:
    """Dense coverage at seq 1024: gate on agreement with the CSR kernel."""
    q, k, v, dout, cu, q2k, row, idx, schedule = _build(1, 1024, 2, 8, 8, seed=34)
    _, c_dq, c_dk, c_dv = _grads_backend(
        "triton", q, k, v, dout, cu, q2k, row, idx, schedule, 8, 1024
    )
    _, s_dq, s_dk, s_dv = _grads_qstat(q, k, v, dout, cu, q2k, row, idx, 8)
    torch.cuda.synchronize()
    for name, a, b in (("dq", s_dq, c_dq), ("dk", s_dk, c_dk), ("dv", s_dv, c_dv)):
        diff = (a.float() - b.float()).abs()
        d, m = diff.max().item(), diff.mean().item()
        assert d < 0.2, f"qstat {name} max deviation from csr backward: {d}"
        assert m < 2e-3, f"qstat {name} mean deviation from csr backward: {m}"
    print("ok qstat-vs-csr agreement at seq=1024")


def test_qstat_dkdv_row_split_matches_unsplit() -> None:
    """Forcing the dK/dV row split must reproduce the unsplit gradients."""
    q, k, v, dout, cu, q2k, row, idx, schedule = _build(1, 1024, 2, 4, 8, seed=37)
    old = os.environ.get("FMHA_SM120_QSTAT_SPLIT_ROWS")
    try:
        os.environ["FMHA_SM120_QSTAT_SPLIT_ROWS"] = "0"
        _, u_dq, u_dk, u_dv = _grads_qstat(q, k, v, dout, cu, q2k, row, idx, 8)
        os.environ["FMHA_SM120_QSTAT_SPLIT_ROWS"] = "128"
        _, s_dq, s_dk, s_dv = _grads_qstat(q, k, v, dout, cu, q2k, row, idx, 8)
    finally:
        if old is None:
            os.environ.pop("FMHA_SM120_QSTAT_SPLIT_ROWS", None)
        else:
            os.environ["FMHA_SM120_QSTAT_SPLIT_ROWS"] = old
    torch.cuda.synchronize()
    # Splitting only re-associates fp32 chunk sums; bf16 storage rounds them.
    torch.testing.assert_close(s_dq.float(), u_dq.float(), rtol=0, atol=0)
    torch.testing.assert_close(s_dk.float(), u_dk.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(s_dv.float(), u_dv.float(), rtol=1e-2, atol=1e-2)
    print("ok qstat dkdv row split")


def test_qstat_fp8_matches_dequant_reference() -> None:
    q, k, v, dout, cu, q2k, row, idx, schedule = _build(2, 512, 2, 4, 4, seed=35)
    k_u8, v_u8, ks, vs = quantize_kv_fp8_scaled(k, v)
    k_deq = (k_u8.view(torch.float8_e4m3fn).float() * ks.unsqueeze(-1)).to(torch.bfloat16)
    v_deq = (v_u8.view(torch.float8_e4m3fn).float() * vs.unsqueeze(0)).to(torch.bfloat16)
    # The kernel quantizes Q per-row to e4m3 before the QK matmul; give the
    # reference the identically quantized Q so the comparison isolates kernel
    # arithmetic rather than absorbing Q-quantization error into tolerances.
    q_amax = q.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    q_used = (
        (q.float() * (448.0 / q_amax)).to(torch.float8_e4m3fn).float() * (q_amax / 448.0)
    ).to(torch.bfloat16)
    r_out, r_dq, r_dk, r_dv = _grads_backend(
        "torch_ref", q_used, k_deq, v_deq, dout, cu, q2k, row, idx, schedule, 4, 512
    )
    qr = q.detach().clone().requires_grad_(True)
    kr = k_deq.detach().clone().requires_grad_(True)
    vr = v_deq.detach().clone().requires_grad_(True)
    out = sparse_attention_qstat_fp8(
        qr, kr, vr, k_u8, v_u8, ks, vs, q2k, row, idx, topk=4, blk_kv=128,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
    )
    out.backward(dout)
    torch.cuda.synchronize()
    # Q quantization is emulated in the reference; the remaining gap is the
    # kernel's P -> e4m3 quantization before PV (<= 2^-4 relative on the
    # attention weights), which the flat reference cannot reproduce. Budget it
    # in atol rather than pretending the paths are bit-comparable. The
    # backward dequantizes K/V once and recomputes scores from unquantized
    # bf16 Q (the reference uses quantized Q), so dq carries the same
    # quantization budget as dk/dv.
    torch.testing.assert_close(out.float(), r_out.float(), rtol=4e-2, atol=8e-2)
    torch.testing.assert_close(qr.grad.float(), r_dq.float(), rtol=1e-1, atol=1.5e-1)
    torch.testing.assert_close(kr.grad.float(), r_dk.float(), rtol=1.5e-1, atol=2.5e-1)
    torch.testing.assert_close(vr.grad.float(), r_dv.float(), rtol=1.5e-1, atol=2.5e-1)
    print("ok qstat fp8")


def test_qstat_mode_through_public_api() -> None:
    """FMHA_SM120_TRITON_MODE=qstat routes sparse_atten_func's autograd path."""
    q, k, v, dout, cu, q2k, row, idx, schedule = _build(1, 512, 2, 4, 4, seed=36)
    old = {
        k_: os.environ.get(k_)
        for k_ in ("FMHA_SM120_BACKEND", "FMHA_SM120_TRITON_MODE", "FMHA_SM120_TRITON_STRICT")
    }
    try:
        os.environ["FMHA_SM120_BACKEND"] = "triton"
        os.environ["FMHA_SM120_TRITON_MODE"] = "qstat"
        os.environ["FMHA_SM120_TRITON_STRICT"] = "1"
        qr = q.detach().clone().requires_grad_(True)
        kr = k.detach().clone().requires_grad_(True)
        vr = v.detach().clone().requires_grad_(True)
        out = sparse_atten_func(
            qr, kr, vr, row, idx, 4, blk_kv=128, causal=True,
            cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=512, max_seqlen_k=512,
            schedule=schedule, q2k_indices=q2k,
        )
        out.backward(dout)
        # forward-only route as well
        with torch.no_grad():
            out2, lse2 = sparse_atten_func(
                q, k, v, row, idx, 4, blk_kv=128, causal=True, return_softmax_lse=True,
                cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=512, max_seqlen_k=512,
                schedule=schedule, q2k_indices=q2k,
            )
    finally:
        for k_, v_ in old.items():
            if v_ is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v_
    torch.cuda.synchronize()
    for name, grad in (("dq", qr.grad), ("dk", kr.grad), ("dv", vr.grad)):
        assert grad is not None and torch.isfinite(grad.float()).all(), name
        assert float(grad.float().abs().sum().item()) > 0.0, f"{name} is zero"
    torch.testing.assert_close(out2.float(), out.detach().float(), rtol=1e-3, atol=1e-3)
    assert lse2.shape == (512, 8)
    print("ok qstat via public API (autograd + forward)")


def test_qstat_fp8_cuda_forward_impl() -> None:
    """FMHA_SM120_QSTAT_IMPL=cuda on the fp8 path: full-e4m3 forward."""
    q, k, v, dout, cu, q2k, row, idx, schedule = _build(1, 512, 2, 4, 4, seed=39)
    k_u8, v_u8, ks, vs = quantize_kv_fp8_scaled(k, v)
    k_deq = (k_u8.view(torch.float8_e4m3fn).float() * ks.unsqueeze(-1)).to(torch.bfloat16)
    v_deq = (v_u8.view(torch.float8_e4m3fn).float() * vs.unsqueeze(0)).to(torch.bfloat16)

    def run() -> tuple[torch.Tensor, ...]:
        qr = q.detach().clone().requires_grad_(True)
        kr = k_deq.detach().clone().requires_grad_(True)
        vr = v_deq.detach().clone().requires_grad_(True)
        out = sparse_attention_qstat_fp8(
            qr, kr, vr, k_u8, v_u8, ks, vs, q2k, row, idx, topk=4, blk_kv=128,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
        )
        out.backward(dout)
        torch.cuda.synchronize()
        return out.detach(), qr.grad, kr.grad, vr.grad

    old = os.environ.get("FMHA_SM120_QSTAT_IMPL")
    try:
        os.environ["FMHA_SM120_QSTAT_IMPL"] = "triton"
        t_out, t_dq, t_dk, t_dv = run()
        os.environ["FMHA_SM120_QSTAT_IMPL"] = "cuda"
        c_out, c_dq, c_dk, c_dv = run()
    finally:
        if old is None:
            os.environ.pop("FMHA_SM120_QSTAT_IMPL", None)
        else:
            os.environ["FMHA_SM120_QSTAT_IMPL"] = old
    # Both impls quantize S/P/V the same way in structure but compute p
    # independently, so e4m3 bucket-boundary flips bound the pointwise gap.
    for name, a, b, mx in (
        ("out", c_out, t_out, 0.15),
        ("dq", c_dq, t_dq, 0.25),
        ("dk", c_dk, t_dk, 0.25),
        ("dv", c_dv, t_dv, 0.25),
    ):
        diff = (a.float() - b.float()).abs()
        assert diff.max().item() < mx and diff.mean().item() < 5e-3, (
            f"{name}: max={diff.max().item()} mean={diff.mean().item()}"
        )
    print("ok qstat fp8 cuda forward impl")


def test_qstat_fp8_grads_impl() -> None:
    """FMHA_SM120_QSTAT_GRADS=fp8: experimental full-e4m3 backward."""
    q, k, v, dout, cu, q2k, row, idx, schedule = _build(1, 512, 2, 4, 4, seed=40)
    k_u8, v_u8, ks, vs = quantize_kv_fp8_scaled(k, v)
    k_deq = (k_u8.view(torch.float8_e4m3fn).float() * ks.unsqueeze(-1)).to(torch.bfloat16)
    v_deq = (v_u8.view(torch.float8_e4m3fn).float() * vs.unsqueeze(0)).to(torch.bfloat16)

    def run() -> tuple[torch.Tensor, ...]:
        qr = q.detach().clone().requires_grad_(True)
        kr = k_deq.detach().clone().requires_grad_(True)
        vr = v_deq.detach().clone().requires_grad_(True)
        out = sparse_attention_qstat_fp8(
            qr, kr, vr, k_u8, v_u8, ks, vs, q2k, row, idx, topk=4, blk_kv=128,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
        )
        out.backward(dout)
        torch.cuda.synchronize()
        return out.detach(), qr.grad, kr.grad, vr.grad

    old_impl = os.environ.get("FMHA_SM120_QSTAT_IMPL")
    old_grads = os.environ.get("FMHA_SM120_QSTAT_GRADS")
    try:
        os.environ["FMHA_SM120_QSTAT_IMPL"] = "cuda"
        os.environ["FMHA_SM120_QSTAT_GRADS"] = "bf16"
        b_out, b_dq, b_dk, b_dv = run()
        os.environ["FMHA_SM120_QSTAT_GRADS"] = "fp8"
        f_out, f_dq, f_dk, f_dv = run()
    finally:
        for name, val in (("FMHA_SM120_QSTAT_IMPL", old_impl),
                          ("FMHA_SM120_QSTAT_GRADS", old_grads)):
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val
    torch.testing.assert_close(f_out.float(), b_out.float())  # fwd identical
    # Gradient quantization band (documented; adoption gated on loss A/B).
    for name, a, b in (("dq", f_dq, b_dq), ("dk", f_dk, b_dk), ("dv", f_dv, b_dv)):
        cos = torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0).item()
        rel = (a.float() - b.float()).abs().mean().item() / b.float().abs().mean().item()
        assert cos > 0.995 and rel < 0.08, f"{name}: cos={cos} rel={rel}"
    print("ok qstat fp8 grads impl (full-e4m3 backward)")


def test_qstat_cuda_forward_impl() -> None:
    """FMHA_SM120_QSTAT_IMPL=cuda: hand-written forward + Triton backward."""
    q, k, v, dout, cu, q2k, row, idx, schedule = _build(1, 1024, 2, 4, 8, seed=38)
    old = os.environ.get("FMHA_SM120_QSTAT_IMPL")
    try:
        os.environ["FMHA_SM120_QSTAT_IMPL"] = "triton"
        t_out, t_dq, t_dk, t_dv = _grads_qstat(q, k, v, dout, cu, q2k, row, idx, 8)
        os.environ["FMHA_SM120_QSTAT_IMPL"] = "cuda"
        c_out, c_dq, c_dk, c_dv = _grads_qstat(q, k, v, dout, cu, q2k, row, idx, 8)
    finally:
        if old is None:
            os.environ.pop("FMHA_SM120_QSTAT_IMPL", None)
        else:
            os.environ["FMHA_SM120_QSTAT_IMPL"] = old
    torch.cuda.synchronize()
    torch.testing.assert_close(c_out.float(), t_out.float(), rtol=2e-2, atol=2e-2)
    # Backward consumes the CUDA forward's out/lse; gradients must stay at the
    # kernel-agreement floor.
    for name, a, b in (("dq", c_dq, t_dq), ("dk", c_dk, t_dk), ("dv", c_dv, t_dv)):
        diff = (a.float() - b.float()).abs()
        assert diff.max().item() < 0.1 and diff.mean().item() < 2e-3, (
            f"{name}: max={diff.max().item()} mean={diff.mean().item()}"
        )
    print("ok qstat cuda forward impl (fwd + mixed backward)")


def main() -> int:
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    test_qstat_matches_reference()
    test_qstat_agrees_with_csr_backward_at_scale()
    test_qstat_dkdv_row_split_matches_unsplit()
    test_qstat_fp8_matches_dequant_reference()
    test_qstat_mode_through_public_api()
    test_qstat_cuda_forward_impl()
    test_qstat_fp8_cuda_forward_impl()
    test_qstat_fp8_grads_impl()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
