# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""JIT loader for the hand-written SM120 qstat forward kernel.

Compiled lazily on first use via torch cpp_extension (cached in
TORCH_EXTENSIONS_DIR). Selected with FMHA_SM120_QSTAT_IMPL=cuda; the Triton
kernels remain the default and the reference.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load

from src.common.arch import target_sm_arch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


_CUDA_CFLAGS = [
    "-O3",
    f"-arch=sm_{target_sm_arch('120')}",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
]


@lru_cache(maxsize=1)
def _ext():
    return load(
        name="qstat_fwd_sm120_ext",
        sources=[os.path.join(_THIS_DIR, "csrc", "qstat_fwd_sm120.cu")],
        extra_cuda_cflags=_CUDA_CFLAGS,
        verbose=False,
    )


@lru_cache(maxsize=1)
def _ext_fp8():
    return load(
        name="qstat_fwd_fp8_sm120_ext",
        sources=[os.path.join(_THIS_DIR, "csrc", "qstat_fwd_fp8_sm120.cu")],
        extra_cuda_cflags=_CUDA_CFLAGS,
        verbose=False,
    )


@lru_cache(maxsize=1)
def _ext_bwd_fp8():
    return load(
        name="qstat_bwd_fp8_sm120_ext",
        sources=[os.path.join(_THIS_DIR, "csrc", "qstat_bwd_fp8_sm120.cu")],
        extra_cuda_cflags=_CUDA_CFLAGS,
        verbose=False,
    )


def qstat_forward_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    union: torch.Tensor,
    counts: torch.Tensor,
    selbits: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    block_t: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, head_q, _ = q.shape
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    out = _ext().qstat_fwd_v3(
        q, k, v, union, counts, selbits, lse,
        int(batch), int(seq_len), int(block_t), float(softmax_scale),
    )
    return out, lse


def qstat_forward_fp8_cuda(
    q: torch.Tensor,
    k_fp8_u8: torch.Tensor,
    v_fp8_u8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    union: torch.Tensor,
    counts: torch.Tensor,
    selbits: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    block_t: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, head_q, _ = q.shape
    # The kernel reads V dim-major (16B cp.async chunk = 16 tokens at one
    # channel); build the transposed view once per call (~a small memcpy).
    v8t = v_fp8_u8.permute(1, 2, 0).contiguous()
    lse = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    out = _ext_fp8().qstat_fwd_fp8v2(
        q, k_fp8_u8, v8t, k_scale, v_scale, union, counts, selbits, lse,
        int(batch), int(seq_len), int(block_t), float(softmax_scale),
    )
    return out, lse


def qstat_backward_fp8_cuda(
    q: torch.Tensor,
    k_fp8_u8: torch.Tensor,
    v_fp8_u8: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dout: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    union: torch.Tensor,
    counts: torch.Tensor,
    selbits: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    row_batch: torch.Tensor,
    row_kv_block: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    block_t: int,
    block_tq: int,
    topk: int,
    softmax_scale: float,
    kv_grad_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-e4m3 backward (FMHA_SM120_QSTAT_GRADS=fp8).

    Gradient MMAs run in e4m3 with per-row dO/dS quantization.  The softmax
    delta is computed twice: delta (raw dO, for the Triton dK/dV fallback
    which uses raw dO) and delta_q (from the quantize-dequantized dO', used
    by the fp8 kernels).  delta_q restores the softmax-gradient shift
    invariance dS = P*(dP - delta) under the quantized gradient field —
    using the raw delta against quantized dp injects a deterministic,
    step-correlated rank-one bias on dq that stalls attention learning
    (val-loss flatline from sparse activation).  Gate on a loss-level A/B
    covering >=5k post-sparse-activation steps.
    """
    ext = _ext_bwd_fp8()
    total_q, head_q, dim = q.shape
    head_kv = k_fp8_u8.shape[1]
    g = head_q // head_kv
    dkdv_fp8_ok = _dkdv_fp8_supported()
    # Row-quantize Q and dO' = dO * v_scale[channel] in one pass.
    q8 = torch.empty((total_q, head_q, dim), device=q.device, dtype=torch.uint8)
    do8 = torch.empty_like(q8)
    qsc = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    dosc = torch.empty_like(qsc)
    ext.qstat_quant_rows(q, dout, v_scale, q8, qsc, do8, dosc, g)
    # dQ (fuses both deltas, consumed by dK/dV).
    k8t = k_fp8_u8.permute(1, 2, 0).contiguous()
    delta = torch.empty((total_q, head_q), device=q.device, dtype=torch.float32)
    delta_q = torch.empty_like(delta)
    dq = ext.qstat_dq_fp8(
        q, k_fp8_u8, k8t, v_fp8_u8, k_scale, v_scale, dout, out, lse,
        union, counts, selbits, delta, delta_q,
        batch, seq_len, block_t, softmax_scale,
    )
    # dK/dV with the deterministic long-row split of the bf16 backward.
    row_counts = k2q_row_ptr[:, 1:] - k2q_row_ptr[:, :-1]
    max_row_count = int(row_counts.max().item()) if row_counts.numel() else 0
    split_rows = int(os.environ.get("FMHA_SM120_QSTAT_SPLIT_ROWS", "2048"))
    nsplit = 1
    if split_rows > 0 and max_row_count > split_rows:
        nsplit = min(8, -(-max_row_count // split_rows))
    if nsplit > 1:
        dk = torch.empty((nsplit, total_q, head_kv, dim), device=q.device, dtype=torch.float32)
        dv = torch.empty_like(dk)
    else:
        dk = torch.empty((total_q, head_kv, dim), device=q.device, dtype=kv_grad_dtype)
        dv = torch.empty_like(dk)
    if dkdv_fp8_ok:
        # fp8 dK/dV computes dp from the quantized dO' (do8) -> needs delta_q.
        ext.qstat_dkdv_fp8(
            q8, qsc, do8, dosc, k_fp8_u8, v_fp8_u8, k_scale, v_scale, lse, delta_q,
            k2q_row_ptr, k2q_q_indices, row_batch, row_kv_block, dk, dv,
            seq_len, block_tq, topk, nsplit, softmax_scale,
        )
    else:
        # Mixed mode for processes where the >48KB opt-in is unavailable:
        # keep the (48KB, mesh-safe) CUDA dQ above and run dK/dV through the
        # Triton kernel. The kernel MUST see the raw e4m3 K/V + scales
        # (KV_FP8=True) so it requantizes Q per row and recomputes p in the
        # exact score field the saved LSE normalizes — running it on
        # dequant-once bf16 K/V against the fp8-field LSE is the softmax
        # field mismatch 53136a0 measured at dq rel-err 7.8 (8x post-merge
        # logit spike) to 2e6 (16x). dp comes from raw dO (GRAD_FP8=False),
        # so the raw `delta` pairs correctly, same as the deployed
        # GRADS=bf16 backward.
        from src.sm120.qstat import _pick_block_t, _qstat_bwd_dkdv_kernel

        total_rows = row_batch.shape[0]
        sub_n = 64
        nsub = 128 // sub_n
        grad_split_stride = (
            total_q * head_kv * dim if nsplit > 1 else 0
        )
        grid_dkdv = (total_rows * nsub, head_kv, nsplit)
        _qstat_bwd_dkdv_kernel[grid_dkdv](
            q, k_fp8_u8, v_fp8_u8, k2q_row_ptr, k2q_q_indices, row_batch,
            row_kv_block, k_scale, v_scale, lse, delta, dout, dk, dv,
            int(grad_split_stride), float(softmax_scale), int(total_q),
            int(total_rows), int(seq_len), int(head_q), int(head_kv),
            int(g), int(topk), True, False,
            NSPLIT=int(nsplit), BLOCK_TQ=int(block_tq), BLK_KV=128,
            SUB_N=int(sub_n), DIM=int(dim), num_warps=8, num_stages=2,
        )
    if nsplit > 1:
        dk = dk.sum(0).to(kv_grad_dtype)
        dv = dv.sum(0).to(kv_grad_dtype)
    return dq, dk, dv


@lru_cache(maxsize=1)
def _dkdv_fp8_supported() -> bool:
    # FMHA_SM120_QSTAT_DKDV_FP8=0 forces the Triton dK/dV fallback (testing /
    # belt-and-suspenders for processes where the probe itself misbehaves).
    if os.environ.get("FMHA_SM120_QSTAT_DKDV_FP8", "auto") == "0":
        return False
    ok = bool(_ext_bwd_fp8().qstat_dkdv_fp8_supported())
    if not ok:
        import warnings

        warnings.warn(
            "qstat dK/dV fp8 kernel cannot opt into >48KB shared memory in "
            "this process (many-fatbin cudaFuncSetAttribute failure); "
            "FMHA_SM120_QSTAT_GRADS=fp8 will run CUDA dQ + Triton dK/dV.",
            stacklevel=2,
        )
    return ok
