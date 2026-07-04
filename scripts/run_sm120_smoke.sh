#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
#
# SM120 (RTX PRO 6000 Blackwell) smoke suite: correctness tests across the
# forward/backward/decode/NVFP4 paths, then short prefill benchmarks.

set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export FMHA_CUDA_ARCH="${FMHA_CUDA_ARCH:-120}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120}"
export FMHA_SM120_BACKEND="${FMHA_SM120_BACKEND:-triton}"
export FMHA_SM120_TRITON_MODE="${FMHA_SM120_TRITON_MODE:-auto}"
export FMHA_SM120_TRITON_STRICT="${FMHA_SM120_TRITON_STRICT:-1}"
export FMHA_SM120_PARTIAL_DTYPE="${FMHA_SM120_PARTIAL_DTYPE:-bf16}"
export MINFER_FMHA_CACHE_DIR="${MINFER_FMHA_CACHE_DIR:-${HOME}/.cache/minfer/fmha_sm120}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions_sm120}"

python tests/sm120/test_port.py
python tests/sm120/test_triton_forward.py
FMHA_SM120_TRITON_MODE=recompute python tests/sm120/test_triton_forward.py
python tests/sm120/test_nvfp4.py
python tests/sm120/test_nvfp4_train.py
FMHA_SM120_PARTIAL_DTYPE=fp32 python tests/sm120/test_triton_backward.py
python tests/sm120/test_autograd_guard.py
python tests/sm120/test_qstat.py
python tests/sm120/test_indexer_fp4.py
python tests/sm120/test_decode.py
FMHA_SM120_DECODE_SPLIT_PAGES=2 python tests/sm120/test_decode.py
python benchmarks/bench_sm120_triton.py --seq 2048 --topk 16 --causal --iters 20 --warmup 5 --modes two_phase,recompute,row
FMHA_SM120_BACKWARD_MODE=csr python benchmarks/bench_sm120_triton_train.py --seq 2048 --topk 16 --causal --iters 8 --warmup 2 --modes two_phase,recompute
