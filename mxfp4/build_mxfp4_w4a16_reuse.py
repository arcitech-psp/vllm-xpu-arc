"""CPU-only build of the candidate-local persistent-scratchpad operator."""
import os
import pathlib

os.environ["MAX_JOBS"] = "1"
compiler = "/opt/intel/oneapi/compiler/2026.1/bin/"
os.environ["CXX"] = compiler + "icpx"
os.environ["PATH"] = compiler + ":/opt/venv/bin:" + os.environ["PATH"]
os.environ["TORCH_XPU_ARCH_LIST"] = ""

from torch.utils.cpp_extension import load
import torch

assert not torch.xpu.is_initialized()
root = pathlib.Path(__file__).resolve().parent
build = root / "mxfp4-w4a16-reuse-build"
build.mkdir(exist_ok=True)
dnnl = "/opt/intel/oneapi/dnnl/2026.0"
load(
    name="vllm_xpu_mxfp4_woq_reuse",
    sources=[str(root / "mxfp4_w4a16_reuse.cpp")],
    extra_include_paths=[dnnl + "/include"],
    extra_cflags=["-O3", "-DNDEBUG", "-fsycl", "-Wno-deprecated-declarations"],
    extra_ldflags=[
        "-fsycl",
        "-ltorch_xpu",
        "-lc10_xpu",
        "-L" + dnnl + "/lib",
        "-Wl,-rpath," + dnnl + "/lib",
        "-ldnnl",
    ],
    with_sycl=False,
    is_python_module=False,
    build_directory=str(build),
    verbose=True,
)
assert not torch.xpu.is_initialized()
print("MXFP4_W4A16_REUSE_COMPILED_NO_XPU_CONTEXT", flush=True)
