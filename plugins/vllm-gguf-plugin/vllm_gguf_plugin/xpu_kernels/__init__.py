# SPDX-License-Identifier: Apache-2.0
"""SYCL GGUF kernels for Intel XPU. Rebuild with build_gguf_xpu.py (icpx matching the SYCL runtime)."""
import os
from pathlib import Path

import torch

SYCL_MOE_TYPES = frozenset({11, 12, 13, 14})  # Q3_K, Q4_K, Q5_K, Q6_K
_loaded = None


def available() -> bool:
    """Load gguf_xpu_moe.so once; VLLM_GGUF_XPU_KERNELS=0 disables it."""
    global _loaded
    if _loaded is None:
        _loaded = False
        if os.environ.get("VLLM_GGUF_XPU_KERNELS", "1") != "0" and hasattr(torch, "xpu") and torch.xpu.is_available():
            library = Path(os.environ.get("VLLM_GGUF_XPU_KERNEL_LIB", Path(__file__).with_name("gguf_xpu_moe.so")))
            try:
                torch.ops.load_library(str(library))
                _register_fake()
                _loaded = True
            except (OSError, RuntimeError):
                _loaded = False
    return _loaded


def _register_fake() -> None:
    from torch.library import register_fake

    @register_fake("gguf_xpu::moe_vec")
    def _moe_vec_fake(x, w, ids, top_k, qtype, rows):
        return torch.empty((x.shape[0] * top_k, rows), dtype=x.dtype, device=x.device)


def moe_vec(x, w, topk_ids, top_k: int, qtype: int, rows: int):
    return torch.ops.gguf_xpu.moe_vec(x, w, topk_ids, top_k, qtype, rows)
