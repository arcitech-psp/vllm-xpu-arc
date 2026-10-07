#!/usr/bin/env python3
"""Check the explicitly selected B70 before starting our vLLM service."""
import json
import math
import os
import subprocess
import sys
if len(sys.argv) > 1 and sys.argv[1] == '--arc-preflight':
    import torch
    if torch.xpu.device_count() != 1:
        raise SystemExit('Exactly one visible XPU is required; verify ZE_AFFINITY_MASK')
    properties = torch.xpu.get_device_properties(0)
    if 'B70' not in properties.name:
        raise SystemExit(f'Expected B70, got {properties.name}; refusing to use another device')
    free, total = torch.xpu.mem_get_info(0)
    util = float(sys.argv[sys.argv.index('--gpu-memory-utilization') + 1])
    print('ARC_XPU_PREFLIGHT ' + json.dumps({'device': properties.name, 'free_bytes': free,
        'total_bytes': total, 'utilization': util, 'required_bytes': math.ceil(total * util)}), flush=True)
    if free < math.ceil(total * util):
        raise SystemExit('B70 has competing or unreleased allocations. Preserve the requested budget; inspect DRM clients.')
    raise SystemExit(0)

# Terminate the preflight child completely, closing its DRM handles, before the
# server/worker is spawned. The parent never imports or initializes torch/XPU.
subprocess.run([sys.executable,__file__,'--arc-preflight',*sys.argv[1:]],check=True)
os.execv('/opt/venv/bin/vllm', ['vllm', 'serve', *sys.argv[1:]])
