#!/usr/bin/env python3
"""Repro + regression test for the mesh-forward 'invalid resource handle' bug.

Simulates what a hybrid multi-GPU forward can do: the thread's current CUDA
device differs from the device the MSA tensors live on. Without a CUDAGuard in
the extension host code, cudaMemsetAsync/kernel launches use a stream from the
tensor's device against the current device's context -> invalid resource
handle. Run from ~/MSA with venv-msa.
"""
import os
import sys

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python" / "fmha_sm100" / "cute"))

import torch  # noqa: E402

from sparse_index_utils import build_k2q_csr  # noqa: E402


def make_q2k(hkv, total, topk, num_blocks, device):
    torch.manual_seed(0)
    q2k = torch.full((hkv, total, topk), -1, dtype=torch.int32, device=device)
    for t in range(total):
        blk = t // 128
        sel = torch.randperm(blk + 1)[:topk]
        sel = torch.sort(sel).values.to(torch.int32)
        q2k[:, t, : len(sel)] = sel.to(device)
    return q2k


def run_case(tensor_dev, current_dev):
    S = 512
    B, hkv, topk, g = 1, 2, 4, 4
    total = B * S
    q2k = make_q2k(hkv, total, topk, S // 128, tensor_dev)
    cu = torch.arange(0, B + 1, device=tensor_dev, dtype=torch.int32) * S
    torch.cuda.set_device(current_dev)   # <-- the mesh-forward hazard
    row, idx, sched = build_k2q_csr(
        q2k, cu, cu, 128, total_k=total, max_seqlen_k=S, max_seqlen_q=S,
        total_rows=B * (S // 128), qhead_per_kv=g, return_schedule=True)
    torch.cuda.synchronize(tensor_dev)
    assert row.device == q2k.device
    print(f"ok: tensors on {tensor_dev}, current device {current_dev}")


import pytest  # noqa: E402


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2+ GPUs")
def test_build_k2q_csr_cross_device():
    run_case("cuda:0", "cuda:0")
    run_case("cuda:0", "cuda:1")
    run_case("cuda:1", "cuda:0")
    print("DEVICE_GUARD_TEST_OK")


if __name__ == "__main__":
    assert torch.cuda.device_count() >= 2, "need 2+ GPUs for this repro"
    test_build_k2q_csr_cross_device()
