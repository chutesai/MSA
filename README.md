# MiniMax Sparse Attention (MSA)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](pyproject.toml)
[![GPU](https://img.shields.io/badge/NVIDIA-SM100-76b900.svg)](#requirements)
[![Stack: CuTe-DSL + Cuda](https://img.shields.io/badge/stack-CuTe--DSL%20%2B%20Cuda-purple.svg)](#stacks)

**MSA** (`fmha_sm100`) ships dense FlashAttention and sparse top-k attention
kernels for **NVIDIA SM100**. Two JIT-compiled stacks
share one Python package:

![MSA architecture](docs/architecture.png)

> Algorithm reference: [MiniMax Sparse Attention paper](docs/MiniMaxSparseAttention.pdf).

| Stack | Path | What it gives you |
|---|---|---|
| **csrc JIT** | `python/fmha_sm100/csrc/` | Dense FMHA (`fmha_sm100`, `fmha_sm100_plan`) + `sparse_topk_select` indexer, compiled from Jinja templates by `jit.py` at runtime. |
| **CuTe-DSL** | `python/fmha_sm100/cute/` | Full sparse attention (forward + paged FP8 decode, BF16 / FP8 / NVFP4 / FP4), compiled at runtime via `cute.compile`. |
| **Bridge** | `python/fmha_sm100/sparse_fmha_adapter.py` | Adapts the `fmha_sm100` API to call `sparse_atten_func` for sparse prefill paths. |

> **License: MIT.** Self-authored files carry `SPDX-License-Identifier: MIT`.
> See [LICENSE](LICENSE) and [NOTICE](NOTICE). Bundled / derived third-party
> code retains its own license — see [Third-party licenses](#third-party-licenses).

## Requirements

- **GPU**: NVIDIA SM100 for the full CuTe-DSL stack; SM120 (RTX PRO 6000
  Blackwell) runs through a Triton backend — see [SM120 support](#sm120-support).
- **Toolchain**: CUDA Toolkit with `nvcc` on `PATH` (or `CUDA_HOME` / `CUDA_PATH` set).
- **Python**: ≥ 3.10.
- **OS**: Linux x86_64 (aarch64 untested; JIT builds may need small Makefile edits on WSL).

Quick sanity check before installing:

```bash
nvcc --version                # expect ≥ 12.x
nvidia-smi --query-gpu=compute_cap --format=csv | grep "10.0"  # confirm SM100
python -c "import sys; print(sys.version_info[:2])"              # ≥ (3, 10)
```

## Using with the `kernels` library

To quickly get started using MSA kernels, you can use the [`kernels` library](https://github.com/huggingface/kernels):

```py
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("MiniMaxAI/msa", version=0)
sparse_atten_func = kernel_module.sparse_atten_func

sparse_atten_func(...)
```

Check out the kernel on the Hugging Face Hub [here](https://huggingface.co/kernels/kernels-staging/msa).

## Install

```bash
# --recursive pulls the CUTLASS submodule (python/fmha_sm100/cutlass/), whose
# headers are required for JIT/AOT compilation. The submodule tracks a CUTLASS
# fork branch that adds SM120 build support on top of the pinned upstream base.
git clone --recursive https://github.com/MiniMax-AI/MSA.git msa
cd msa
# If you cloned without --recursive:
#   git submodule update --init --recursive
pip install .           # standard install (works from a wheel too)
# or
pip install -e .        # editable install for development
```

This pulls in the CuTe-DSL stack via `nvidia-cutlass-dsl` and `quack-kernels`;
the csrc kernels are JIT-compiled at first import from sources shipped inside
the package.

## Verify

Run a small CUDA smoke test. **The first run JIT-compiles `sparse_topk_select`,
which takes 30 s – a few minutes on a cold nvcc cache** — this is normal, not
a hang. Subsequent runs hit the JIT cache and finish in seconds.

```bash
python tests/smoke/test_sparse_topk_forced.py
```

## Usage

```python
import torch
from fmha_sm100 import fmha_sm100, fmha_sm100_plan, sparse_topk_select

# Page size and top-k for the sparse prefill path.
page_size, topk = 128, 16

# Dense proxy pass: compute per-block max score from a cheap Q slice.
proxy_plan = fmha_sm100_plan(
    qo_lens, kv_lens, proxy_q.shape[1],
    num_kv_heads=1,
    page_size=page_size,
    output_maxscore=True,
)
_, max_score = fmha_sm100(
    proxy_q, proxy_k_pages, proxy_v_pages, proxy_plan,
    kv_indices=kv_indices,
    output_o=False,
    output_maxscore=True,
)

# max_score -> sparse KV block indexes.
kv_block_indexes = sparse_topk_select(
    max_score.contiguous(), topk, num_valid_pages=num_pages,
)

# Sparse attention with the selected blocks.
sparse_plan = fmha_sm100_plan(
    qo_lens, kv_lens, q.shape[1],
    num_kv_heads=k_pages.shape[1],
    page_size=page_size,
    kv_block_num=topk,
)
out, _ = fmha_sm100(
    q, k_pages, v_pages, sparse_plan,
    kv_indices=kv_indices,
    kv_block_indexes=kv_block_indexes,
)
```

For block-sparse prefill with CSR metadata, the FP4 indexer, NVFP4 K/V, and
the paged FP8 decode wrapper, see the **CuTe-DSL deep dive**:

- [`python/fmha_sm100/cute/README.md`](python/fmha_sm100/cute/README.md)

## Test

```bash
# Fast smoke tests.
python -m pytest tests/smoke -q

# API and end-to-end integration tests.
python -m pytest tests/integration -q
python tests/integration/test_proxy_kv_e2e.py

# Large regression suites.
python tests/regression/test_correctness.py
python tests/regression/test_sparse_attn.py

# CuTe-DSL forward-only sparse attention.
cd python/fmha_sm100/cute
python -m pytest test_sparse_atten.py -q

# SM120 Triton backend (requires an SM120-class GPU).
python -m pytest tests/sm120 -q
```

## SM120 support

SM120 GPUs (RTX PRO 6000 Blackwell, compute capability 12.0) lack the
tcgen05/TMEM instructions the SM100 CuTe-DSL kernels are built on. On these
devices the public sparse attention API dispatches to a Triton backend in
[`python/fmha_sm100/cute/src/sm120/`](python/fmha_sm100/cute/src/sm120/)
instead:

- **Prefill** (`sparse_atten_func`): BF16 Q/K/V forward and backward (custom
  autograd), plus forward-only BF16 Q with FP8 E4M3 K/V.
- **NVFP4 K/V prefill** (`sparse_atten_nvfp4_kv_func`): packed NVFP4 K/V with
  128x4 scales; `sparse_atten_nvfp4_kv_train_func` adds a training path that
  quantizes BF16 K/V on the fly and backpropagates to the logical tensors
  (the on-the-fly quantizer requires a Transformer Engine install with NVFP4
  support; the packed-input paths do not).
- **Decode** (`sparse_decode_atten_func`, `SparseDecodePagedAttentionWrapper`):
  paged FP8 decode.
- **FP4 indexer** (`fp4_indexer_block_scores`): on SM120 the block-score pass
  dispatches to a Triton kernel using native block-scaled FP4 MMA
  (`src/sm120/fp4_indexer.py`; MXFP4, public scale layout). Scoring is dense
  and compute-bound, so FP4 pays fully here (~3x over a BF16 cuBLAS scoring
  baseline at seq 8192) while only affecting top-k page *selection*.
- A pure-PyTorch reference (`src/sm120/reference.py`) backs every path as a
  correctness oracle and functional fallback.

For training, `FMHA_SM120_TRITON_MODE=qstat` selects the Q-stationary kernels
(`src/sm120/qstat.py`): single-pass online softmax with the GQA group folded
into the MMA M dimension, and an atomics-free backward (deterministic
gradients). Each query tile iterates the union of blocks its tokens selected,
so throughput depends on selection overlap between neighboring tokens: on
realistic sink+local patterns it is ~4-6x faster than the CSR modes (and the
forward reaches roughly 60% of peak BF16 throughput on RTX PRO 6000), while
on adversarial uncorrelated-random selections the CSR modes remain faster.
Constraints: causal, fixed-length batches, no paged KV, D=128,
`qhead_per_kv` in {1,2,4,8,16}. `sparse_attention_qstat_fp8` additionally
takes pre-quantized E4M3 K/V with per-token/per-channel scales.

The csrc JIT stack (dense FMHA, `sparse_topk_select`, plan kernel) compiles
for SM120 with `FMHA_CUDA_ARCH=120`; the plan kernel's shared-memory layout is
automatically sized down for the smaller opt-in shared memory per block.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `FMHA_SM120_BACKEND` | `auto` | `auto`/`triton`/`torch_ref`/`off`. `auto` selects Triton on SM120-class devices and is a no-op elsewhere. |
| `FMHA_CUDA_ARCH` | `100a` | Target arch for the csrc JIT and CUDA extensions, e.g. `120`. |
| `CUTE_DSL_ARCH` | unset | Overrides the CuTe-DSL target (e.g. `sm_120`); falls back to `FMHA_CUDA_ARCH`. |
| `FMHA_SM120_TRITON_MODE` | `auto` | Prefill strategy: `auto`/`two_phase`/`chunked`/`recompute`/`row`/`qstat`. `auto` switches from two-phase to recompute when partial buffers would exceed `FMHA_SM120_MAX_PARTIAL_MIB` (default 1024). `qstat` selects the Q-stationary training kernels (below). |
| `FMHA_SM120_BACKWARD` | `triton` | `triton` or `torch_ref` backward for the autograd path. |
| `FMHA_SM120_TRITON_STRICT` | `0` | `1` raises instead of falling back to the torch reference on unsupported cases. |
| `FMHA_SM120_DECODE_SPLIT_PAGES` | `0` | Pages per split for decode split-KV. Unset, sparse decode runs unsplit; small-batch decode can gain ~30% from `2`-`4`. Dense decode over more than 32 pages always splits to bound kernel unrolling. |

Run the full SM120 suite (correctness plus short benchmarks) with:

```bash
bash scripts/run_sm120_smoke.sh
```

## Benchmark

`benchmarks/bench_sparse_attention_ops.py` covers dense prefill, paged
prefill, sparse prefill, dense decode, paged decode, sparse decode, in
`fp8` and `bf16` (`nvfp4` is sparse-prefill only).

```bash
python benchmarks/bench_sparse_attention_ops.py --help     # full flag list
```

Common invocations (output is TSV):

| Goal | Command |
|---|---|
| FP8 full sweep | `python benchmarks/bench_sparse_attention_ops.py --dtype fp8 --sections all --output_mode o -o /tmp/msa_fp8.tsv` |
| BF16 full sweep | `python benchmarks/bench_sparse_attention_ops.py --dtype bf16 --sections all --output_mode o -o /tmp/msa_bf16.tsv` |
| NVFP4 sparse prefill | `python benchmarks/bench_sparse_attention_ops.py --dtype nvfp4 --sections sparse_prefill --output_mode o -o /tmp/msa_nvfp4.tsv` |
| Quick CI smoke | `python benchmarks/bench_sparse_attention_ops.py --dtype fp8 --sections prefill,decode,sparse_decode --seqs 8192,16384 --tp 1,4 --decode-k 8192,131072 --decode-b 32 --dry-run-ms 50 --repeat-ms 200 -o /tmp/msa_smoke.tsv` |
| Output-mode checks (dense/paged) | `--output_mode maxscore` or `--output_mode full` |

## Layout

```
python/fmha_sm100/                  Python package
  __init__.py                       Public re-exports (lazy for the CuTe-DSL stack)
  api.py                            fmha_sm100 / fmha_sm100_plan / sparse_topk_select
  jit.py                            Runtime JIT (nvcc + ninja) for the csrc stack
  sparse.py                         Lazy shim that loads the cute/ stack
  sparse_fmha_adapter.py            Bridge: fmha_sm100 API → sparse_atten_func
  csrc/                             CUDA kernels + Jinja templates (JIT-compiled)
    include/                        Vendored FlashInfer / CUTLASS-derived / TRT-LLM headers
  cutlass/                          NVIDIA CUTLASS git submodule (include/ + tools/util/include/)
  cute/                             CuTe-DSL sparse attention (loaded via sys.path)
    src/sm100/                      SM100 CuTe-DSL kernels
    src/sm120/                      SM120 Triton backend + torch reference
tests/                              Correctness tests
  smoke/  integration/  regression/  sm120/
scripts/                            Warmup + cache-management + smoke helpers
benchmarks/                         bench_sparse_attention_ops.py, bench_sm120_*.py
```

## Stacks

- **csrc JIT** — dense FlashAttention, page KV, and `sparse_topk_select`
  indexer. Compiled at runtime from `csrc/*.cu.jinja` plus
  `csrc/include/`. Public entry: `fmha_sm100.plan → run`.
- **CuTe-DSL** — block-sparse prefill, FP8 / NVFP4 / FP4 quantization, paged
  FP8 decode (`SparseDecodePagedAttentionWrapper`), FP4 block-score indexer.
  Public entry: `fmha_sm100.sparse_atten_func`,
  `fmha_sm100.sparse_decode_atten_func`, `fmha_sm100.fp4_indexer_block_scores`.
- **Bridge** — `sparse_fmha_plan` / `sparse_fmha` adapt the dense-API call
  site to the sparse backend for prefill paths; useful when you already
  drive the dense kernel and want a one-line swap to sparse.

## Third-party licenses

`fmha_sm100` bundles, derives from, or depends on the third-party components
below. Each retains its original license; this section summarizes them.
Authoritative text is shipped with each component.

### Vendored / derived source (shipped in this repo)

| Component | License | Where |
|---|---|---|
| **NVIDIA CUTLASS** | BSD-3-Clause | Git submodule at `python/fmha_sm100/cutlass/` (provides `include/` + `tools/util/include/`), plus BSD-3-tagged headers under `python/fmha_sm100/csrc/include/`. The SM100 MMA descriptor encodings in `python/fmha_sm100/cute/src/common/mma_sm100_desc.py` mirror CUTLASS hardware descriptors. Copyright (c) 2017–2025 NVIDIA CORPORATION & AFFILIATES. |
| **FlashInfer** | Apache-2.0 | Headers and sources under `python/fmha_sm100/csrc/` and `python/fmha_sm100/csrc/include/` that carry a `Copyright (c) <year> by FlashInfer team` line (e.g. `allocator.h`, `exception.h`, `utils.cuh`, `cutlass_utils.cuh`, `fmha_cutlass_sm100.cuh`, `sparse_topk_select.cuh`, `plan.cuh`, `sm100_fmha_reduction.hpp`, `tvm_ffi_utils.h`). Project: <https://github.com/flashinfer-ai/flashinfer>. |
| **NVIDIA TensorRT-LLM + NAVER Corp (CLOVA)** | Apache-2.0 | Portions of `python/fmha_sm100/csrc/include/sparse_topk_select.cuh` — `indexerTopK` histogram-step + insertion-sort derived from `tensorrt_llm/cpp/tensorrt_llm/kernels/indexerTopK.cu`. Copyright (c) 2019–2026 NVIDIA CORPORATION; Copyright (c) 2021 NAVER Corp. The per-file header in `sparse_topk_select.cuh` includes a function-level provenance map. |

### Runtime dependencies (installed via pip)

| Package | Upstream | License |
|---|---|---|
| `quack-kernels` | <https://github.com/Dao-AILab/quack> | Apache-2.0 |
| `nvidia-cutlass-dsl` | NVIDIA CUTLASS Python DSL | NVIDIA / BSD-3-Clause (see package) |
| `apache-tvm-ffi` | Apache TVM FFI | Apache-2.0 |
| `cuda-python` | NVIDIA | NVIDIA / see package |
| `torch` | <https://github.com/pytorch/pytorch> | BSD-3-Clause |
| `jinja2` | <https://github.com/pallets/jinja> | BSD-3-Clause |
| `ninja` | <https://github.com/ninja-build/ninja> | Apache-2.0 |
| `pybind11` | <https://github.com/pybind/pybind11> | BSD-3-Clause |

The exact license of each installed package is distributed with that package;
consult its metadata (`pip show <pkg>`) for the authoritative text.

## Citation

If MSA helps your research, please cite it. (BibTeX entry coming once the
companion paper / technical report has a stable identifier — placeholder.)
The algorithmic reference is shipped at
[`docs/MiniMaxSparseAttention.pdf`](docs/MiniMaxSparseAttention.pdf).

```bibtex
@software{msa2026,
  title  = {MiniMax Sparse Attention (MSA): FlashAttention and block-sparse
            attention kernels for NVIDIA SM100},
  author = {{MiniMax}},
  year   = {2026},
  url    = {https://github.com/MiniMax-AI/MSA}
}
```

## Contributing

Issues and PRs welcome on the
[issue tracker](https://github.com/MiniMax-AI/MSA/issues). For kernel or
runtime-contract changes, open an issue first to align on the public
surface — `fmha_sm100.api`, `fmha_sm100.sparse` and
`cute.interface` are the stable entry points; everything else
is internal and may change without notice.
