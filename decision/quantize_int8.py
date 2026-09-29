"""INT8 (W8A8) compressed-tensors build of a BF16 Qwen3.5-family text checkpoint, for vLLM on Intel XPU.

Weights: per-output-channel symmetric INT8 (scale = amax / 127, round to nearest), stored as int8 + float32 weight_scale.
Activations: dynamic per-token symmetric INT8 in the config, so vLLM serves it as W8A8 INT8.
Scope (SCOPE env):
  mlp  - MLP gate/up/down only (the parts that carry ~2/3 of the weights); everything else BF16.
  all  - MLP + full-attention q/k/v/o + Gated DeltaNet in_proj_qkv/in_proj_z/out_proj.  (Mintelica INT8 uses `all`.)
Always BF16: embeddings / tied lm_head, norms, conv1d, A_log, dt_bias, in_proj_a / in_proj_b (32-row gates).
Tensor-level (no model load, no calibration data needed); names and shards follow the source checkpoint.
usage: SCOPE=all python3 quantize_int8.py SRC OUT
"""
import json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, OUT = sys.argv[1:3]
SCOPE = os.environ.get('SCOPE', 'mlp')
PAT = {'mlp': r'\.mlp\.(gate|up|down)_proj\.weight$',
       'all': r'(\.mlp\.(gate|up|down)_proj|\.self_attn\.[qkvo]_proj|\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj))\.weight$'}[SCOPE]
QMAX = 127.0   # symmetric int8
os.makedirs(OUT, exist_ok=True)
idx_path = os.path.join(SRC, 'model.safetensors.index.json')
shards = sorted(set(json.load(open(idx_path))['weight_map'].values())) if os.path.exists(idx_path) else ['model.safetensors']
weight_map, n_q, err_sum, t0 = {}, 0, 0.0, time.time()
for shard in shards:
    out = {}
    with safe_open(os.path.join(SRC, shard), 'pt') as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if re.search(PAT, k) and t.dim() == 2:
                w = t.float()
                scale = (w.abs().amax(dim=1, keepdim=True) / QMAX).clamp(min=1e-12)
                q = torch.round(w / scale).clamp(-QMAX, QMAX).to(torch.int8)
                err_sum += ((q.float() * scale - w).pow(2).mean() / w.pow(2).mean().clamp(min=1e-20)).item()
                out[k] = q
                out[k[:-len('weight')] + 'weight_scale'] = scale.to(torch.float32)
                n_q += 1
            else:
                out[k] = t
    save_file(out, os.path.join(OUT, shard), metadata={'format': 'pt'})
    for k in out: weight_map[k] = shard
    print(f'{shard}: done ({time.time() - t0:.0f}s)', flush=True)
if len(shards) > 1:
    json.dump({'metadata': {}, 'weight_map': weight_map}, open(os.path.join(OUT, 'model.safetensors.index.json'), 'w'), indent=1)
ignore = ['lm_head', 're:.*embed_tokens.*', 're:.*norm.*', 're:.*in_proj_a.*', 're:.*in_proj_b.*']
if SCOPE == 'mlp':
    ignore += ['re:.*self_attn.*', 're:.*linear_attn.*']
cfg = json.load(open(os.path.join(SRC, 'config.json')))
cfg['quantization_config'] = {
    'quant_method': 'compressed-tensors', 'format': 'int-quantized', 'quantization_status': 'compressed',
    'config_groups': {'group_0': {'targets': ['Linear'], 'format': 'int-quantized',
        'weights': {'num_bits': 8, 'type': 'int', 'symmetric': True, 'strategy': 'channel', 'dynamic': False, 'observer': 'minmax'},
        'input_activations': {'num_bits': 8, 'type': 'int', 'symmetric': True, 'strategy': 'token', 'dynamic': True, 'observer': None}}},
    'ignore': ignore, 'kv_cache_scheme': None, 'global_compression_ratio': None, 'version': '0.17.1'}
json.dump(cfg, open(os.path.join(OUT, 'config.json'), 'w'), indent=1)
for fn in os.listdir(SRC):
    if fn.endswith(('.json', '.jinja', '.txt')) and fn not in ('config.json', 'model.safetensors.index.json'):
        shutil.copy2(os.path.join(SRC, fn), os.path.join(OUT, fn))
print(f'DONE {OUT}: scope={SCOPE}, {n_q} matrices INT8, mean relative weight MSE {err_sum / max(1, n_q):.2e}', flush=True)
