#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Compare SM120 Triton paged FP8 decode against the torch reference backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))

from interface import SparseDecodePagedAttentionWrapper, sparse_decode_atten_func  # noqa: E402


def _make_inputs(*, batch: int, seqlen_q: int, kv_tokens: int, head_kv: int, seed: int):
    torch.manual_seed(seed)
    device = "cuda"
    blk_kv = 128
    dim = 128
    qhead_per_kv = 16
    head_q = head_kv * qhead_per_kv
    page_count = (kv_tokens + blk_kv - 1) // blk_kv
    q = torch.randn(batch * seqlen_q, head_q, dim, device=device).to(torch.float8_e4m3fn)
    k = torch.randn(batch, page_count, head_kv, blk_kv, dim, device=device).to(torch.float8_e4m3fn)
    v = torch.randn(batch, page_count, head_kv, blk_kv, dim, device=device).to(torch.float8_e4m3fn)
    tail = kv_tokens - (page_count - 1) * blk_kv
    if tail < blk_kv:
        k[:, -1, :, tail:, :] = 0
        v[:, -1, :, tail:, :] = 0
    k_pages = k.reshape(batch * page_count, head_kv, blk_kv, dim).contiguous()
    v_pages = v.reshape(batch * page_count, head_kv, blk_kv, dim).contiguous()
    page_table = torch.arange(batch * page_count, device=device, dtype=torch.int32).view(batch, page_count)
    seqused_k = torch.full((batch,), kv_tokens, dtype=torch.int32, device=device)
    return q, k_pages, v_pages, page_table, seqused_k


def _make_sparse_q2k(*, head_kv: int, total_q: int, page_count: int, topk: int, device: str) -> torch.Tensor:
    pages = torch.arange(page_count, device=device, dtype=torch.int32)
    if topk <= page_count:
        selected = pages[-topk:]
    else:
        selected = torch.cat(
            [pages, torch.full((topk - page_count,), -1, device=device, dtype=torch.int32)]
        )
    return selected.view(1, 1, topk).expand(head_kv, total_q, topk).contiguous()


def _run_function_case(*, sparse: bool, seqlen_q: int, kv_tokens: int, seed: int) -> None:
    batch = 3
    head_kv = 2
    blk_kv = 128
    q, k, v, page_table, seqused_k = _make_inputs(
        batch=batch,
        seqlen_q=seqlen_q,
        kv_tokens=kv_tokens,
        head_kv=head_kv,
        seed=seed,
    )
    q2k = None
    if sparse:
        q2k = _make_sparse_q2k(
            head_kv=head_kv,
            total_q=q.shape[0],
            page_count=page_table.shape[1],
            topk=min(4, page_table.shape[1]),
            device="cuda",
        )
    kwargs = dict(
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=seqlen_q,
        max_seqlen_k=kv_tokens,
        blk_kv=blk_kv,
        causal=True,
        return_softmax_lse=True,
    )
    old_backend = os.environ.get("FMHA_SM120_BACKEND")
    old_strict = os.environ.get("FMHA_SM120_TRITON_STRICT")
    try:
        os.environ["FMHA_SM120_BACKEND"] = "torch_ref"
        ref_out, ref_lse = sparse_decode_atten_func(q, k, v, q2k, **kwargs)
        os.environ["FMHA_SM120_BACKEND"] = "triton"
        os.environ["FMHA_SM120_TRITON_STRICT"] = "1"
        tri_out, tri_lse = sparse_decode_atten_func(q, k, v, q2k, **kwargs)
    finally:
        if old_backend is None:
            os.environ.pop("FMHA_SM120_BACKEND", None)
        else:
            os.environ["FMHA_SM120_BACKEND"] = old_backend
        if old_strict is None:
            os.environ.pop("FMHA_SM120_TRITON_STRICT", None)
        else:
            os.environ["FMHA_SM120_TRITON_STRICT"] = old_strict
    torch.cuda.synchronize()
    torch.testing.assert_close(tri_lse, ref_lse, rtol=2e-3, atol=3e-3)
    torch.testing.assert_close(tri_out.float(), ref_out.float(), rtol=2e-1, atol=2e-1)
    print(
        f"ok function sparse={sparse} seqlen_q={seqlen_q} kv_tokens={kv_tokens} "
        f"out={tuple(tri_out.shape)}"
    )


def _run_wrapper_dense_case() -> None:
    q, k, v, page_table, seqused_k = _make_inputs(
        batch=2,
        seqlen_q=1,
        kv_tokens=384,
        head_kv=1,
        seed=44,
    )
    wrapper = SparseDecodePagedAttentionWrapper(blk_kv=128, causal=True)
    wrapper.plan(
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=1,
        max_seqlen_k=384,
        q2k_indices=None,
        num_qo_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_dim=q.shape[2],
    )
    old_backend = os.environ.get("FMHA_SM120_BACKEND")
    try:
        os.environ["FMHA_SM120_BACKEND"] = "triton"
        out, lse = wrapper.run(q, k, v, return_softmax_lse=True)
        out_buf = torch.empty_like(out)
        lse_buf = torch.empty_like(lse)
        out2, lse2 = wrapper.run(
            q,
            k,
            v,
            return_softmax_lse=True,
            out=out_buf,
            lse=lse_buf,
        )
    finally:
        if old_backend is None:
            os.environ.pop("FMHA_SM120_BACKEND", None)
        else:
            os.environ["FMHA_SM120_BACKEND"] = old_backend
    torch.cuda.synchronize()
    assert out.shape == q.shape
    assert lse.shape == q.shape[:2]
    assert out2.data_ptr() == out_buf.data_ptr()
    assert lse2.data_ptr() == lse_buf.data_ptr()
    torch.testing.assert_close(out2, out, rtol=0, atol=0)
    torch.testing.assert_close(lse2, lse, rtol=0, atol=0)
    print(f"ok wrapper dense out={tuple(out.shape)}")


def test_decode_function() -> None:
    _run_function_case(sparse=False, seqlen_q=1, kv_tokens=256, seed=31)
    _run_function_case(sparse=False, seqlen_q=4, kv_tokens=384, seed=32)
    _run_function_case(sparse=True, seqlen_q=2, kv_tokens=512, seed=33)


def test_decode_wrapper_dense() -> None:
    _run_wrapper_dense_case()


def main() -> int:
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    test_decode_function()
    test_decode_wrapper_dense()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
