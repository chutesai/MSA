#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Compare SM120 packed NVFP4 K/V path against a BF16 dequant reference."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "fmha_sm100" / "cute"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from interface import sparse_atten_func, sparse_atten_nvfp4_kv_func  # noqa: E402
from sparse_index_utils import build_k2q_csr  # noqa: E402
from test_triton_forward import _make_q2k, _pack_identity_pages  # noqa: E402


_FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _make_scales(required_rows: int, *, device: str) -> torch.Tensor:
    padded_rows = ((int(required_rows) + 127) // 128) * 128
    padded_cols = 8
    # E4M3FN encoding of +1.0: sign=0, exp=bias=7, mant=0.
    return torch.full((padded_rows, padded_cols), 0x38, dtype=torch.uint8, device=device)


def _pack_from_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    low = nibbles[..., 0::2]
    high = nibbles[..., 1::2]
    return (low | (high << 4)).contiguous().to(torch.uint8)


def _dequant_from_nibbles(nibbles: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    values = _FP4_VALUES.to(device=nibbles.device)[nibbles.long()]
    return (values * global_scale.float()).to(torch.bfloat16)


def _run_case(*, paged: bool, seed: int) -> None:
    torch.manual_seed(seed)
    device = "cuda"
    blk_kv = 128
    dim = 128
    head_kv = 2
    qhead_per_kv = 4
    head_q = head_kv * qhead_per_kv
    batch_lens = (256, 384)
    topk = 4
    cu = [0]
    for length in batch_lens:
        cu.append(cu[-1] + int(length))
    cu_q = torch.tensor(cu, device=device, dtype=torch.int32)
    cu_k = torch.tensor(cu, device=device, dtype=torch.int32)
    total_q = cu[-1]
    total_k = cu[-1]
    max_len = max(batch_lens)
    total_rows = sum((length + blk_kv - 1) // blk_kv for length in batch_lens)
    q = torch.randn(total_q, head_q, dim, device=device, dtype=torch.bfloat16)
    k_global = torch.tensor([0.125], device=device, dtype=torch.float32)
    v_global = torch.tensor([0.125], device=device, dtype=torch.float32)

    k_nibbles = torch.randint(0, 16, (total_k, head_kv, dim), device=device, dtype=torch.uint8)
    v_nibbles = torch.randint(0, 16, (total_k, head_kv, dim), device=device, dtype=torch.uint8)
    k_dense = _dequant_from_nibbles(k_nibbles, k_global)
    v_dense = _dequant_from_nibbles(v_nibbles, v_global)
    k_pack = _pack_from_nibbles(k_nibbles)
    v_pack = _pack_from_nibbles(v_nibbles)

    page_table = None
    k_ref = k_dense
    v_ref = v_dense
    k_call = k_pack
    v_call = v_pack
    if paged:
        k_ref, v_ref, page_table = _pack_identity_pages(k_dense, v_dense, batch_lens, blk_kv=blk_kv)
        k_call, v_call, page_table2 = _pack_identity_pages(k_pack, v_pack, batch_lens, blk_kv=blk_kv)
        torch.testing.assert_close(page_table2, page_table)
        required_scale_rows = int(k_call.shape[0]) * head_kv * blk_kv
    else:
        required_scale_rows = total_k * head_kv
    k_scale = _make_scales(required_scale_rows, device=device)
    v_scale = _make_scales(required_scale_rows, device=device)

    q2k = _make_q2k(head_kv, batch_lens, topk, blk_kv=blk_kv, device=device)
    row, idx, schedule = build_k2q_csr(
        q2k,
        cu_q,
        cu_k,
        blk_kv,
        total_k=total_k,
        max_seqlen_k=max_len,
        max_seqlen_q=max_len,
        total_rows=total_rows,
        qhead_per_kv=qhead_per_kv,
        return_schedule=True,
    )
    kwargs = dict(
        blk_kv=blk_kv,
        causal=True,
        return_softmax_lse=True,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_len,
        max_seqlen_k=max_len,
        schedule=schedule,
    )
    old_backend = os.environ.get("FMHA_SM120_BACKEND")
    try:
        os.environ["FMHA_SM120_BACKEND"] = "torch_ref"
        ref_out, ref_lse = sparse_atten_func(
            q,
            k_ref,
            v_ref,
            row,
            idx,
            topk,
            page_table=page_table,
            q2k_indices=q2k,
            **kwargs,
        )
        os.environ["FMHA_SM120_BACKEND"] = "triton"
        tri_out, tri_lse = sparse_atten_nvfp4_kv_func(
            q,
            k_call,
            v_call,
            k_scale,
            v_scale,
            k_global,
            v_global,
            row,
            idx,
            topk,
            page_table=page_table,
            **kwargs,
        )
    finally:
        if old_backend is None:
            os.environ.pop("FMHA_SM120_BACKEND", None)
        else:
            os.environ["FMHA_SM120_BACKEND"] = old_backend

    torch.cuda.synchronize()
    torch.testing.assert_close(tri_lse, ref_lse, rtol=5e-3, atol=8e-3)
    torch.testing.assert_close(tri_out.float(), ref_out.float(), rtol=4e-2, atol=4e-2)
    print(f"ok nvfp4 paged={paged} out={tuple(tri_out.shape)}")


def test_nvfp4_flat() -> None:
    _run_case(paged=False, seed=101)


def test_nvfp4_paged() -> None:
    _run_case(paged=True, seed=102)


def main() -> int:
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    test_nvfp4_flat()
    test_nvfp4_paged()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
