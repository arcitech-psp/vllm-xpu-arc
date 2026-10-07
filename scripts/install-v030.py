#!/usr/bin/env python3
"""Install and compile our pinned v0.30 rebase, with no XPU context."""
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import torch

assert not torch.xpu.is_initialized(), 'CPU build must not initialize XPU'
root = Path(__file__).resolve().parents[1]
site = Path(sysconfig.get_paths()['purelib'])
os.environ.setdefault('MAX_JOBS', '1')
for name in ['0001-vllm-v0.30.0-ced6857-xpu-extras.patch', '0003-holo4-xpu-boundaries.patch']:
    subprocess.run(['patch','--batch','--forward','-p1','-d',str(site),'-i',str(root/'experimental/v0.30-rebase/patches'/name)],check=True)
for name in ['vllm_xpu_draft_lmhead_int4.py','vllm_xpu_draft_mtp_int4.py']:
    shutil.copy2(root/'patches'/name,site/'vllm/model_executor/models'/name)
try:
    importlib.metadata.version('gguf')
except importlib.metadata.PackageNotFoundError:
    subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','gguf==0.19.0'],check=True)
plugin = root/'plugins/vllm-gguf-plugin'
env = {**os.environ, 'VLLM_GGUF_PLUGIN_SKIP_EXT':'1','VLLM_GGUF_BUILD_CUDA':'0'}
subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--no-deps','--no-build-isolation',str(plugin)],env=env,check=True)
subprocess.run([sys.executable,str(plugin/'vllm_gguf_plugin/xpu_kernels/build_gguf_xpu.py')],check=True)
# The wheel is installed before native compilation; install the new .so there.
shutil.copy2(plugin/'vllm_gguf_plugin/xpu_kernels/gguf_xpu_moe.so',site/'vllm_gguf_plugin/xpu_kernels/gguf_xpu_moe.so')
subprocess.run([sys.executable,str(root/'adaptive/build_kernel.py')],check=True)
subprocess.run([sys.executable,str(root/'mxfp4/build_mxfp4_w4a16.py'),'--onednn','installed'],check=True)
assert not torch.xpu.is_initialized(), 'CPU build initialized XPU'
runtime_site = Path('/opt/vllm-xpu-arc/runtime-site')
runtime_site.mkdir(parents=True, exist_ok=True)
for entry in site.iterdir():
    if entry.name in {'vllm','vllm_gguf_plugin','gguf'} or (
        entry.name.endswith('.dist-info') and entry.name.startswith(('vllm_gguf_plugin-', 'gguf-'))
    ):
        shutil.copytree(entry,runtime_site/entry.name,dirs_exist_ok=True)
print('ARC_V030_CPU_BUILD_OK',flush=True)
