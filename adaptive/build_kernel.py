"""Build a separate SYCL operator on CPU; do not replace vendor libraries."""
import os
from pathlib import Path

os.environ.setdefault("MAX_JOBS", "2")
os.environ["CXX"] = "/opt/intel/oneapi/compiler/2026.0/bin/icpx"
os.environ["PATH"] = "/opt/intel/oneapi/compiler/2026.0/bin:" + os.environ["PATH"]
# Compile portable SPIR64 on CPU; do not initialize a competing XPU context
# to discover device targets while the candidate model is running.
os.environ["TORCH_XPU_ARCH_LIST"] = ""
import torch
assert not torch.xpu.is_initialized()
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
(root / "kernel-build").mkdir(exist_ok=True)
load(name="vllm_xpu_adaptive_gdn", sources=[str(root / "kernel/adaptive_gdn.cpp")],
     extra_include_paths=[str(root / "kernel")],
     extra_cflags=["-O2", "-fsycl", "-Wno-deprecated-declarations"],
     # icpx handles device linking once. The installed PyTorch separate SYCL
     # dlink path feeds compressed device bitcode back to llvm-link and fails.
     extra_ldflags=["-fsycl", "-ltorch_xpu", "-lc10_xpu"], with_sycl=False, is_python_module=False,
     build_directory=str(root / "kernel-build"), verbose=True)
assert not torch.xpu.is_initialized()
print("ADAPTIVE_GDN_BUILD_OK")
