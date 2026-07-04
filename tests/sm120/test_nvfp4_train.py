#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Exercise the SM120 NVFP4 K/V training path forward and backward.

The core test drives sparse_attention_nvfp4_kv_triton_autograd with
synthetically packed FP4 data, so it needs no Transformer Engine install.
The reference is the differentiable torch backend on the dequantized K/V:
in-kernel dequant and host-side dequant compute identical values, so outputs
agree up to kernel arithmetic and gradients (straight-through at the
quantizer boundary) up to atomic-add accumulation noise.

A second test covers the public sparse_atten_nvfp4_kv_train_func, which
quantizes BF16 K/V on the fly via Transformer Engine; it is skipped when TE
is not installed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from interface import sparse_atten_func, sparse_atten_nvfp4_kv_train_func  # noqa: E402
from sparse_index_utils import build_k2q_csr  # noqa: E402
from src.sm120.atten_triton import (  # noqa: E402
    _dequant_nvfp4_to_bf16,
    sparse_attention_nvfp4_kv_triton_autograd,
)
from test_nvfp4 import _make_scales, _pack_from_nibbles  # noqa: E402
from test_triton_forward import _make_q2k  # noqa: E402


def _build_case(*, seq: int, topk: int, seed: int):
    torch.manual_seed(seed)
    device = "cuda"
    blk_kv = 128
    dim = 128
    head_kv = 2
    qhead_per_kv = 4
    head_q = head_kv * qhead_per_kv
    q = torch.randn(seq, head_q, dim, device=device, dtype=torch.bfloat16)
    dout = torch.randn(seq, head_q, dim, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, seq], device=device, dtype=torch.int32)
    q2k = _make_q2k(head_kv, (seq,), topk, blk_kv=blk_kv, device=device)
    row, idx, schedule = build_k2q_csr(
        q2k,
        cu,
        cu,
        blk_kv,
        total_k=seq,
        max_seqlen_k=seq,
        max_seqlen_q=seq,
        total_rows=(seq + blk_kv - 1) // blk_kv,
        qhead_per_kv=qhead_per_kv,
        return_schedule=True,
    )
    return q, dout, cu, q2k, row, idx, schedule, blk_kv, dim, head_kv


def _reference_grads(q, k_deq, v_deq, row, idx, q2k, schedule, cu, *, topk, blk_kv, causal, dout, seq):
    os.environ["FMHA_SM120_BACKEND"] = "torch_ref"
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k_deq.detach().clone().requires_grad_(True)
    v_ref = v_deq.detach().clone().requires_grad_(True)
    ref_out = sparse_atten_func(
        q_ref,
        k_ref,
        v_ref,
        row,
        idx,
        topk,
        blk_kv=blk_kv,
        causal=causal,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        schedule=schedule,
        q2k_indices=q2k,
    )
    (ref_out.float() * dout.float()).sum().backward()
    return ref_out, q_ref.grad, k_ref.grad, v_ref.grad


def _assert_grads_close(tri, ref):
    tri_out, dq_t, dk_t, dv_t = tri
    ref_out, dq_r, dk_r, dv_r = ref
    torch.testing.assert_close(tri_out.float(), ref_out.float(), rtol=4e-2, atol=4e-2)
    torch.testing.assert_close(dq_t.float(), dq_r.float(), rtol=8e-2, atol=5e-2)
    torch.testing.assert_close(dk_t.float(), dk_r.float(), rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(dv_t.float(), dv_r.float(), rtol=1e-1, atol=1e-1)
    for name, grad in (("dq", dq_t), ("dk", dk_t), ("dv", dv_t)):
        assert torch.isfinite(grad.float()).all(), f"{name} has non-finite values"
        assert float(grad.float().abs().sum().item()) > 0.0, f"{name} is zero"


def run_autograd_case(*, seq: int, topk: int, causal: bool, seed: int) -> None:
    q, dout, cu, q2k, row, idx, schedule, blk_kv, dim, head_kv = _build_case(
        seq=seq, topk=topk, seed=seed
    )
    k_nibbles = torch.randint(0, 16, (seq, head_kv, dim), device="cuda", dtype=torch.uint8)
    v_nibbles = torch.randint(0, 16, (seq, head_kv, dim), device="cuda", dtype=torch.uint8)
    k_packed = _pack_from_nibbles(k_nibbles)
    v_packed = _pack_from_nibbles(v_nibbles)
    k_scale = _make_scales(seq * head_kv, device="cuda")
    v_scale = _make_scales(seq * head_kv, device="cuda")
    # Small global scale keeps random FP4 logits in a realistic numeric range.
    k_global = torch.tensor([0.125], device="cuda", dtype=torch.float32)
    v_global = torch.tensor([0.125], device="cuda", dtype=torch.float32)
    k_deq = _dequant_nvfp4_to_bf16(k_packed, k_scale, k_global)
    v_deq = _dequant_nvfp4_to_bf16(v_packed, v_scale, v_global)

    old_backend = os.environ.get("FMHA_SM120_BACKEND")
    try:
        ref = _reference_grads(
            q, k_deq, v_deq, row, idx, q2k, schedule, cu,
            topk=topk, blk_kv=blk_kv, causal=causal, dout=dout, seq=seq,
        )
        os.environ["FMHA_SM120_BACKEND"] = "triton"
        q_tri = q.detach().clone().requires_grad_(True)
        k_tri = k_deq.detach().clone().requires_grad_(True)
        v_tri = v_deq.detach().clone().requires_grad_(True)
        tri_out = sparse_attention_nvfp4_kv_triton_autograd(
            q_tri,
            k_tri,
            v_tri,
            k_packed,
            v_packed,
            k_scale,
            v_scale,
            k_global,
            v_global,
            row,
            idx,
            q2k_indices=q2k,
            topk=topk,
            blk_kv=blk_kv,
            causal=causal,
            softmax_scale=dim ** -0.5,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            page_table=None,
        )
        (tri_out.float() * dout.float()).sum().backward()
    finally:
        if old_backend is None:
            os.environ.pop("FMHA_SM120_BACKEND", None)
        else:
            os.environ["FMHA_SM120_BACKEND"] = old_backend
    torch.cuda.synchronize()
    _assert_grads_close((tri_out, q_tri.grad, k_tri.grad, v_tri.grad), ref)
    print(f"ok nvfp4_autograd seq={seq} topk={topk} causal={causal}")


def test_nvfp4_autograd_matches_dequant_reference() -> None:
    run_autograd_case(seq=256, topk=4, causal=False, seed=21)
    run_autograd_case(seq=384, topk=4, causal=True, seed=22)


def test_nvfp4_train_func_end_to_end() -> None:
    """Public train func: TE-quantized K/V. Skipped when TE is unavailable."""
    try:
        from quantize import quantize_kv_bf16_to_nvfp4_128x4
        seq, topk, causal, seed = 256, 4, True, 23
        q, dout, cu, q2k, row, idx, schedule, blk_kv, dim, head_kv = _build_case(
            seq=seq, topk=topk, seed=seed
        )
        k = torch.randn(seq, head_kv, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(seq, head_kv, dim, device="cuda", dtype=torch.bfloat16)
        k_q, v_q = quantize_kv_bf16_to_nvfp4_128x4(k, v)
    except (ImportError, RuntimeError) as exc:
        print(f"SKIP nvfp4_train_func: {exc}")
        return
    k_deq = _dequant_nvfp4_to_bf16(k_q.data, k_q.scale_128x4, k_q.global_scale)
    v_deq = _dequant_nvfp4_to_bf16(v_q.data, v_q.scale_128x4, v_q.global_scale)
    old_backend = os.environ.get("FMHA_SM120_BACKEND")
    try:
        ref = _reference_grads(
            q, k_deq, v_deq, row, idx, q2k, schedule, cu,
            topk=topk, blk_kv=blk_kv, causal=causal, dout=dout, seq=seq,
        )
        os.environ["FMHA_SM120_BACKEND"] = "triton"
        q_tri = q.detach().clone().requires_grad_(True)
        k_tri = k.detach().clone().requires_grad_(True)
        v_tri = v.detach().clone().requires_grad_(True)
        tri_out = sparse_atten_nvfp4_kv_train_func(
            q_tri,
            k_tri,
            v_tri,
            row,
            idx,
            topk,
            q2k_indices=q2k,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=seq,
            max_seqlen_k=seq,
            blk_kv=blk_kv,
            causal=causal,
        )
        (tri_out.float() * dout.float()).sum().backward()
    finally:
        if old_backend is None:
            os.environ.pop("FMHA_SM120_BACKEND", None)
        else:
            os.environ["FMHA_SM120_BACKEND"] = old_backend
    torch.cuda.synchronize()
    _assert_grads_close((tri_out, q_tri.grad, k_tri.grad, v_tri.grad), ref)
    print(f"ok nvfp4_train_func seq={seq} topk={topk} causal={causal}")


def main() -> int:
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    test_nvfp4_autograd_matches_dequant_reference()
    test_nvfp4_train_func_end_to_end()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
