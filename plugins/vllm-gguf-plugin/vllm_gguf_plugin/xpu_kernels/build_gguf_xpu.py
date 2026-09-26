"""Build the GGUF XPU MoE kernel as a torch op library (icpx 2026.0 = the container's SYCL runtime)."""
import os
import pathlib

compiler = '/opt/intel/oneapi/compiler/2026.0'
os.environ['CXX'] = compiler + '/bin/icpx'
os.environ['PATH'] = compiler + '/bin:' + os.environ['PATH']
os.environ['MAX_JOBS'] = '4'
os.environ['TORCH_XPU_ARCH_LIST'] = ''

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

root = pathlib.Path(__file__).resolve().parent
build = root / 'kernel-build'
build.mkdir(exist_ok=True)
load(name='gguf_xpu_moe', sources=[str(root / 'gguf_xpu_moe.cpp')],
     extra_cflags=['-O3', '-fsycl', '-Wno-deprecated-declarations'],
     extra_ldflags=['-fsycl', '-ltorch_xpu', '-lc10_xpu'],
     with_sycl=False, is_python_module=False, build_directory=str(build), verbose=False)
assert not torch.xpu.is_initialized()
print('BUILT', build / 'gguf_xpu_moe.so')
