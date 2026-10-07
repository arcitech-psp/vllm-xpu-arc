#!/usr/bin/env python3
"""CPU-only installed-image identity, patch and native-library ABI checks."""
import importlib.metadata
import hashlib
import json
from pathlib import Path
import sysconfig
import torch
import vllm
from vllm import envs

assert not torch.xpu.is_initialized()
site = Path(sysconfig.get_paths()['purelib'])
assert vllm.__version__.startswith('0.30.0'),vllm.__version__
assert envs.VLLM_XPU_INC_WNA16_BACKEND in {'auto','ark','w4a16','w4a8'}
markers = {
    'vllm/v1/attention/backends/gdn_attn.py':'B70_MTP_PARTIAL_FINAL_GROUP',
    'vllm/v1/worker/gpu/sample/logprob.py':'ARC_XPU_BF16_TOPK',
    'vllm/v1/worker/xpu_worker.py':'ARC_XPU_MEMORY',
    'vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py':'ARC_W4A16_BF16_BOUNDARY',
    'vllm/model_executor/models/qwen3_5_mtp.py':'VLLM_XPU_DRAFT_LMHEAD_INT4',
}
for path,marker in markers.items():
    assert marker in (site/path).read_text(),path
    compile((site/path).read_text(),path,'exec')
    if (Path('/source')/path).is_file():
        assert (site/path).read_bytes()==(Path('/source')/path).read_bytes(),path+' source/image mismatch'
for name in ['vllm_xpu_draft_lmhead_int4.py','vllm_xpu_draft_mtp_int4.py']:
    assert (site/'vllm/model_executor/models'/name).read_bytes()==(Path('/repo/patches')/name).read_bytes(),name
entrypoint = Path('/opt/vllm-xpu-arc/scripts/holo4-entrypoint.py')
assert entrypoint.read_bytes()==Path('/repo/scripts/holo4-entrypoint.py').read_bytes(),'entrypoint source/image mismatch'
libraries = [
    site/'vllm_gguf_plugin/xpu_kernels/gguf_xpu_moe.so',
    Path('/opt/vllm-xpu-arc/adaptive/kernel-build/vllm_xpu_adaptive_gdn.so'),
    Path('/opt/vllm-xpu-arc/mxfp4/mxfp4-w4a16-build/vllm_xpu_mxfp4_woq.so'),
]
for library in libraries:
    assert library.is_file(),str(library)
    torch.ops.load_library(str(library))
assert not torch.xpu.is_initialized()
print(json.dumps({'vllm':vllm.__version__,'torch':torch.__version__,
    'xpu_kernels':importlib.metadata.version('vllm-xpu-kernels'),
    'gguf':importlib.metadata.version('gguf'),'patches':list(markers.values()),
    'inc_backend':envs.VLLM_XPU_INC_WNA16_BACKEND,
    'entrypoint_sha256':hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
    'native_libraries_loaded':3,'xpu_initialized':torch.xpu.is_initialized()},indent=2))
