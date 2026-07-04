# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""SM120 (RTX PRO 6000 Blackwell) sparse attention backends.

SM120 lacks the tcgen05/TMEM instructions the SM100 CuTe kernels are built
on, so this package provides Triton kernels (``atten_triton``) plus a plain
PyTorch oracle (``reference``) behind the same CSR varlen contract.
``interface.py`` routes to them based on device capability and
FMHA_SM120_BACKEND; keep this module import-free so the reference backend
stays usable without a Triton install.
"""
