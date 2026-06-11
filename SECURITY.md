# Security Policy

## Reporting a vulnerability

If you discover a security issue in **fmha_sm100 / MiniMax Sparse
Attention (MSA)**, please report it privately. **Do not open a public
GitHub issue** for security-sensitive reports.

Report via one of these channels:

1. **GitHub private vulnerability reporting** (preferred): use
   <https://github.com/MiniMax-AI/MSA/security/advisories/new>. This
   creates a private draft security advisory that only maintainers can
   see.
2. **Email**: `model@minimax.io`. Use a descriptive subject
   line (`[MSA] <short summary>`). Please do not include exploit
   payloads in the initial email — we will respond with a PGP key or
   a private issue tracker link to receive details.

Please include:

- Affected version(s) (commit SHA, tag, or PyPI version)
- A clear description of the issue and its impact
- A minimal reproducer (a Python snippet, a model + input shape, etc.)
- Whether you intend to disclose publicly and on what timeline

## Supported versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes (current dev)  |
| < 0.1   | No                 |

Only the latest minor release receives security fixes. Older versions
will not be patched.

## Embargo policy

- We acknowledge new reports within **3 business days**.
- We aim to ship a fix or mitigation within **30 days** of confirmation
  for high-severity issues, and **90 days** for moderate / low issues.
- We follow **coordinated disclosure**: we ask reporters to keep the
  issue private until we publish a fix and an advisory. We will
  negotiate the disclosure timeline case by case.
- Once a fix is released, the public advisory will credit the reporter
  (unless they prefer to remain anonymous).

## Scope

In-scope reports include, but are not limited to:

- **CUDA kernel safety** — out-of-bounds memory access, illegal memory
  access, race conditions in the JIT-compiled csrc kernels or in the
  CuTe-DSL sparse attention / indexer kernels that lead to a wrong
  output, a kernel fault, or a privilege escalation on the host.
- **Python memory / type confusion** — issues in `fmha_sm100.api`,
  `fmha_sm100.jit`, `fmha_sm100.sparse`, or the `sparse_fmha_adapter`
  that lead to segfault, OOB, or arbitrary code execution.
- **JIT compiler invocation** — issues in the runtime command-line
  composition that compiles user-controlled input to `nvcc` / `cute.compile`.
- **Supply chain** — compromised wheels, malicious upstream merges
  in vendored CUTLASS / FlashInfer / TensorRT-LLM headers, or
  typosquat dependencies in `pyproject.toml` / `requirements.txt`.

Out of scope:

- Issues in upstream dependencies (NVIDIA CUTLASS, FlashInfer,
  TensorRT-LLM, PyTorch, NVIDIA CUTLASS DSL, Apache TVM FFI, etc.)
  — please report those to the upstream projects first; we will
  help coordinate if asked.
- Performance regressions without a correctness or safety impact.
- Denial of service via oversized CUDA allocations on a host the
  attacker does not control.

## Acknowledgements

We are grateful to the security community. Reporters who follow this
policy will be credited in the corresponding advisory.
