# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Target-architecture selection for runtime-compiled kernels.

Everything defaults to SM100. FMHA_CUDA_ARCH (e.g. ``120``) retargets the
JIT-compiled CUDA extensions, and CUTE_DSL_ARCH (e.g. ``sm_120``) additionally
overrides the CuTe DSL target.
"""

from __future__ import annotations

import os


def _normalize(arch: object, default: str) -> str:
    arch = str(arch).strip().lower().removeprefix("sm_").removeprefix("compute_")
    return arch or default


def target_sm_arch(default: str = "100") -> str:
    """Numeric target such as ``"100"`` or ``"120"``, from FMHA_CUDA_ARCH."""
    return _normalize(os.environ.get("FMHA_CUDA_ARCH", default), default)


def target_cute_arch(default: str = "sm_100") -> str:
    """CuTe DSL target such as ``"sm_100"``; CUTE_DSL_ARCH wins over FMHA_CUDA_ARCH."""
    arch = os.environ.get("CUTE_DSL_ARCH") or os.environ.get("FMHA_CUDA_ARCH") or default
    return "sm_" + _normalize(arch, default.removeprefix("sm_"))


def nvcc_arch_flag() -> str:
    """``-arch=sm_XXX`` flag for torch cpp_extension builds."""
    return f"-arch=sm_{target_sm_arch()}"


def cuda_home() -> str:
    from torch.utils.cpp_extension import CUDA_HOME

    return os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or CUDA_HOME or "/usr/local/cuda"
