#!/usr/bin/env python3
"""Read only metadata and safetensors headers; never load or edit weights."""
import argparse
import json
from pathlib import Path
import struct

p=argparse.ArgumentParser()
p.add_argument('model',type=Path)
a=p.parse_args()
config=json.loads((a.model/'config.json').read_text())
qc=config.get('quantization_config') or {}
if qc.get('quant_method') != 'auto-round':
    raise SystemExit('Expected native auto-round metadata; inspect export before serving')
if qc.get('bits') != 4 or qc.get('group_size') != 128 or qc.get('sym') is not True:
    raise SystemExit('Expected symmetric INT4 group 128 export')
if qc.get('packing_format','auto_round:auto_gptq') != 'auto_round:auto_gptq':
    raise SystemExit('Expected auto_round:auto_gptq packing')
if config.get('model_type') != 'qwen3_5' or any('Moe' in v for v in config.get('architectures',[])):
    raise SystemExit('Expected the dense qwen3_5 vision architecture')
index=a.model/'model.safetensors.index.json'
files = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
headers={}
for name in files:
    path=(a.model/name).resolve()
    if not path.is_file():
        raise SystemExit('Indexed shard missing: '+name)
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        if length > 128*1024*1024:
            raise SystemExit('Unexpectedly large safetensors header: '+name)
        for key,value in json.loads(f.read(length)).items():
            if key != '__metadata__':
                headers[key]=value
mtp={k:v for k,v in headers.items() if (k.startswith('mtp.') or '.mtp.' in k) and k.endswith('.weight')}
if not mtp or not any(k.endswith('fc.weight') for k in mtp):
    raise SystemExit('Grafted BF16 MTP weights are not present; wait for Claude to finish the graft')
if any(v['dtype']!='BF16' for v in mtp.values()):
    raise SystemExit('Grafted MTP weights must remain BF16 for our draft build')
preserved={k:v for k,v in headers.items() if k.endswith('.weight') and (
    'visual.' in k or k.endswith('lm_head.weight') or '.in_proj_a.' in k or '.in_proj_b.' in k)}
if not preserved or any(v['dtype']!='BF16' for v in preserved.values()):
    raise SystemExit('Visual/GDN a,b/LM-head BF16 preservation check failed')
print(json.dumps({'quant_method':qc['quant_method'],'bits':4,'group_size':128,'sym':True,
    'packing_format':qc.get('packing_format','auto_round:auto_gptq'),
    'shards':len(files),'mtp_bf16_weight_tensors':len(mtp),'preserved_bf16_weight_tensors':len(preserved)},indent=2))
