#!/usr/bin/env python3
"""Bounded Holo4 operational gates; raw evidence is retained per request."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time
from zoneinfo import ZoneInfo

import requests
from PIL import Image, ImageDraw
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--endpoint', default='http://127.0.0.1:8000')
p.add_argument('--tokenizer', default='/models/holo4')
p.add_argument('--output', type=Path, default=Path('/evidence'))
p.add_argument('--template', default='/templates/chat_template_fast.jinja')
p.add_argument('--repetitions', type=int, default=3)
p.add_argument('--free-marker', type=Path, default=Path('/B70-FREE'))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
tokenizer.chat_template = Path(a.template).read_text()
records = []
expected = ['identity', 'KV pool', 'chat', 'vision', 'tool call parses', 'needle100k',
            'decode C1 4096', 'decode C1 65536', 'decode C1 122880',
            'decode C3 4096', 'decode C3 65536', 'decode C3 122880',
            'JEV LoRA loads and answers', 'MTP partial 128K boundary', 'MTP acceptance nonzero']

def deadline():
    if not a.free_marker.is_file():
        raise RuntimeError('B70-FREE absent: inference is gated')
    now = datetime.now(ZoneInfo('America/Edmonton'))
    if now.date().isoformat() > '2026-10-07' or (now.date().isoformat() == '2026-10-07' and now.hour >= 4):
        raise RuntimeError('04:00 experiment deadline reached')

def get(path):
    deadline()
    r = requests.get(a.endpoint + path, timeout=30)
    r.raise_for_status()
    return r

def metrics(label):
    body = get('/metrics').text
    (a.output / f'{label}.prom').write_text(body)
    values = {}
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if 'le' not in sample.labels:
                values[sample.name] = values.get(sample.name, 0) + sample.value
    return values

def chat(label, messages, model='holo4-27b', **kwargs):
    deadline()
    request = {'model': model, 'messages': messages, 'temperature': 0,
               'max_tokens': 256, 'stream': True, 'stream_options': {'include_usage': True}, **kwargs}
    (a.output / f'{label}.request.json').write_text(json.dumps(request, ensure_ascii=False))
    start = time.monotonic()
    first = None
    chunks = []
    output = ''
    reasoning = ''
    usage = None
    with requests.post(a.endpoint + '/v1/chat/completions', json=request, stream=True, timeout=(30, 1800)) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1):
            deadline()
            if not line.startswith(b'data: ') or line == b'data: [DONE]':
                continue
            chunk = json.loads(line[6:])
            chunks.append(chunk)
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                content = delta.get('content') or ''
                if content and first is None:
                    first = time.monotonic()
                output += content
                reasoning += delta.get('reasoning_content') or delta.get('reasoning') or ''
    end = time.monotonic()
    record = {'label': label, 'ttft_seconds': None if first is None else first-start,
              'total_seconds': end-start, 'after_first_delta_seconds': None if first is None else end-first,
              'first_delta_monotonic':first,'finished_monotonic':end,
              'usage': usage, 'output': output, 'reasoning': reasoning, 'chunks': chunks}
    if usage and first and not reasoning:
        record['client_decode_tokens_s'] = (usage['completion_tokens']-1) / max(end-first, 1e-6)
    (a.output / f'{label}.response.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record

def exact_prompt(target, nonce, needle=False):
    query = 'Write a Python class with many useful methods and explanatory comments.'
    if needle:
        query = 'What is the value on the ARCHIVE_NEEDLE line? Return only that value.'
    header = f'Archive run {nonce}. Read the reference text below.\n'
    fill = tokenizer.encode(' A plain reference line about ordinary buildings.\n', add_special_tokens=False)
    count = max(target // len(fill), 1)
    # Fit the fully rendered chat prompt, not an estimated word count.
    for _ in range(8):
        body_ids = (fill * (count+5))[:max(1, count)] if count > target//2 else fill * count
        body = tokenizer.decode(body_ids)
        if needle:
            mid = len(body)//2
            body = body[:mid] + f'\nARCHIVE_NEEDLE={nonce}\n' + body[mid:]
        messages = [{'role': 'user', 'content': header+body+'\n'+query}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        difference = target - len(ids)
        if difference == 0:
            return messages
        # Switch to token-budget fill after the initial estimate.
        count = len(tokenizer.encode(body, add_special_tokens=False)) + difference
        if needle:
            count -= len(tokenizer.encode(f'\nARCHIVE_NEEDLE={nonce}\n', add_special_tokens=False))
    if abs(target-len(ids)) > 32:
        raise RuntimeError(f'Prompt length calibration failed: {len(ids)} vs {target}')
    return messages

def add(name, passed, details, status=None):
    records.append({'gate': name, 'status': status or ('PASS' if passed else 'FAIL'), 'details': details})
    (a.output / 'gates.json').write_text(json.dumps(records, indent=2))
    lines = ['# Holo4 gates', '', 'Raw request, response, timing and metrics files are in evidence/.', '',
             'Client decode rate is (completion tokens - 1) / time after first visible delta; TTFT is separate.',
             'A synthetic image checks color, shape and spatial recognition; it is a bounded vision smoke.', '']
    for row in records:
        lines += [f"- **{row['gate']}: {row['status']}** — {json.dumps(row['details'])}"]
    for name in expected:
        if not any(r['gate']==name for r in records):
            lines += [f'- **{name}: NOT RUN**']
    (a.output.parent / 'GATES.md').write_text('\n'.join(lines)+'\n')

def exception_hook(kind, value, traceback):
    add('runner exception', False, {'type':kind.__name__, 'error':str(value)})
    sys.__excepthook__(kind,value,traceback)
sys.excepthook = exception_hook

models = get('/v1/models').json()
(a.output / 'models.json').write_text(json.dumps(models, indent=2))
base = metrics('initial')
add('identity', any(m['id']=='holo4-27b' for m in models['data']), models)
if not any(m['id']=='holo4-27b' and m.get('root')=='/models/holo4'
           and m.get('max_model_len')==131072 for m in models['data']):
    raise RuntimeError('Endpoint identity/root/context does not match our Holo4 deployment')
cache = None
for family in text_string_to_metric_families((a.output/'initial.prom').read_text()):
    for sample in family.samples:
        if sample.name.endswith('cache_config_info') and 'kv_cache_size_tokens' in sample.labels:
            cache = sample.labels
if cache:
    capacity = float(cache.get('kv_cache_max_concurrency','0'))
    add('KV pool',capacity >= 1,{'labels':cache,'full_131072_sequence_capacity_floor':math.floor(capacity),
        'note':'Allocated pool capacity reported by vLLM; C3 overlap is measured separately at each supported length.'})
else:
    add('KV pool',False,{'error':'Cache pool metric unavailable; inspect startup logs'})

r = chat('chat', [{'role':'user','content':'What is 19 + 23? Reply with only the number.'}], max_tokens=32,logprobs=True,top_logprobs=5)
add('chat', r['output'].strip().rstrip('.')=='42' and not r['reasoning'], {'output':r['output'], 'usage':r['usage'], 'thinking_default_off':not bool(r['reasoning']), 'top_logprobs_requested':5})

image = Image.new('RGB', (640, 400), 'white')
d = ImageDraw.Draw(image)
d.rectangle((35, 100, 205, 300), fill='blue')
d.ellipse((240, 110, 420, 290), fill='purple')
d.polygon([(450,300),(610,300),(530,80)], fill='orange')
image.save(a.output / 'vision-input.png')
buffer = io.BytesIO()
image.save(buffer, format='PNG')
uri = 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
r = chat('vision', [{'role':'user','content':[{'type':'image_url','image_url':{'url':uri}},
    {'type':'text','text':'Describe the three shapes from left to right, giving each color and shape.'}]}], max_tokens=128)
out = r['output'].lower()
add('vision', all(w in out for w in ['blue','rectangl','purple','circl','orange','triangl'])
    and out.index('blue') < out.index('purple') < out.index('orange'), {'output':r['output']})

tools = [{'type':'function','function':{'name':'get_weather','description':'Get weather for a city.',
    'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
tool_request = {'model':'holo4-27b',
    'messages':[{'role':'user','content':'Use get_weather to check the weather in Edmonton.'}],
    'tools':tools,'tool_choice':'auto','temperature':0,'max_tokens':128}
(a.output/'tool.request.json').write_text(json.dumps(tool_request,indent=2))
response = requests.post(a.endpoint+'/v1/chat/completions', json=tool_request, timeout=180)
response.raise_for_status()
result = response.json()
(a.output/'tool.response.json').write_text(json.dumps(result,indent=2))
calls = result['choices'][0]['message'].get('tool_calls') or []
city = json.loads(calls[0]['function']['arguments']).get('city','') if calls else ''
passed = bool(calls) and calls[0]['function']['name']=='get_weather' and isinstance(city,str) and city.strip().lower().startswith('edmonton')
add('tool call parses', passed, result)

nonce = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:16]
r = chat('needle100k', exact_prompt(100000, nonce, needle=True), max_tokens=64)
add('needle100k', nonce in r['output'] and r['usage']['prompt_tokens'] >= 99968,
    {'output':r['output'], 'expected':nonce, 'usage':r['usage'], 'ttft_seconds':r['ttft_seconds']})

for target in [4096, 65536, 122880]:
    rows = []
    for rep in range(a.repetitions):
        nonce = hashlib.sha256(f'{target}-{rep}-{time.time_ns()}'.encode()).hexdigest()[:16]
        before = metrics(f'c1-{target}-{rep}-before')
        if before.get('vllm:num_requests_running',0) or before.get('vllm:num_requests_waiting',0):
            raise RuntimeError('Server is busy before C1 measurement; preserve the active request')
        row = chat(f'c1-{target}-{rep}', exact_prompt(target, nonce), max_tokens=512,
                   min_tokens=512, ignore_eos=True)
        after = metrics(f'c1-{target}-{rep}-after')
        key = 'vllm:request_decode_time_seconds_sum'
        duration = after.get(key,0)-before.get(key,0)
        count = after.get('vllm:request_decode_time_seconds_count',0)-before.get('vllm:request_decode_time_seconds_count',0)
        row['isolated_server_request_count_delta'] = count
        if duration > 0 and count == 1:
            row['server_decode_seconds'] = duration
            row['server_decode_tokens_s'] = (row['usage']['completion_tokens']-1)/duration
        (a.output/f'c1-{target}-{rep}.measurement.json').write_text(json.dumps(row,indent=2))
        rows.append(row)
    add(f'decode C1 {target}', all(r['usage']['completion_tokens']==512 and not r['reasoning'] and r['isolated_server_request_count_delta']==1 for r in rows),
        {'prompt_tokens':[r['usage']['prompt_tokens'] for r in rows],
         'median_client_decode_tokens_s':statistics.median(r['client_decode_tokens_s'] for r in rows),
         'server_decode_tokens_s':[r.get('server_decode_tokens_s') for r in rows],
         'ttft_seconds':[r['ttft_seconds'] for r in rows], 'n':len(rows)})

for target in [4096,65536,122880]:
    if cache and float(cache['kv_cache_size_tokens']) < 3*(target+512):
        add(f'decode C3 {target}',False,
            {'required_token_capacity':3*(target+512),'pool_token_capacity':cache['kv_cache_size_tokens'],
             'note':'Three full requests cannot fit this allocated pool; no C3 speed claim at this length.'},status='NOT FIT')
        continue
    def concurrent_request(i):
        nonce = hashlib.sha256(f'c3-{target}-{i}-{time.time_ns()}'.encode()).hexdigest()[:16]
        return chat(f'c3-{target}-{i}', exact_prompt(target,nonce), max_tokens=512,min_tokens=512,ignore_eos=True)
    before = metrics(f'c3-{target}-before')
    if before.get('vllm:num_requests_running',0) or before.get('vllm:num_requests_waiting',0):
        raise RuntimeError('Server is busy before C3 measurement; preserve the active request')
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(concurrent_request, range(3)))
    elapsed = time.monotonic()-start
    after = metrics(f'c3-{target}-after')
    overlap = min(r['finished_monotonic'] for r in rows)-max(r['first_delta_monotonic'] for r in rows)
    add(f'decode C3 {target}', all(r['usage']['completion_tokens']==512 and not r['reasoning'] for r in rows) and overlap > 0,
        {'prompt_tokens':[r['usage']['prompt_tokens'] for r in rows],
         'per_chat_client_decode_tokens_s':[r.get('client_decode_tokens_s') for r in rows],
         'ttft_seconds':[r['ttft_seconds'] for r in rows],
         'aggregate_end_to_end_tokens_s':sum(r['usage']['completion_tokens'] for r in rows)/elapsed,
         'all_three_decode_overlap_seconds':max(overlap,0),
         'note':'Aggregate includes prefill; it is not decode throughput.'})

r = chat('jev', [{'role':'user','content':'Choose one word: yes or no. Is 2 + 2 equal to 4?'}], model='jev-decision', max_tokens=64)
add('JEV LoRA loads and answers', bool(r['output'].strip()), {'output':r['output'], 'usage':r['usage']})

deadline()
fill = tokenizer.encode(' A plain reference line about ordinary buildings.\n',add_special_tokens=False)
prompt = (fill * (131066//len(fill)+1))[:131066]
payload = {'model':'holo4-27b','prompt':prompt,'max_tokens':6,'min_tokens':6,'ignore_eos':True,'temperature':0}
(a.output/'boundary.request.json').write_text(json.dumps(payload))
response = requests.post(a.endpoint+'/v1/completions',json=payload,timeout=1800)
response.raise_for_status()
result = response.json()
(a.output/'boundary.response.json').write_text(json.dumps(result,indent=2))
add('MTP partial 128K boundary',result['usage']['prompt_tokens']==131066 and result['usage']['completion_tokens']==6,
    {'usage':result['usage'],'finish_reason':result['choices'][0]['finish_reason']})
final = metrics('final')
draft = sum(v-base.get(k,0) for k,v in final.items() if k.endswith('draft_tokens_total'))
accepted = sum(v-base.get(k,0) for k,v in final.items() if k.endswith('accepted_tokens_total'))
add('MTP acceptance nonzero', draft > 0 and accepted > 0,
    {'draft_tokens':draft,'accepted_tokens':accepted,'acceptance':accepted/draft if draft else None})
raise SystemExit(0 if all(r['status'] in ['PASS','NOT FIT'] for r in records)
    and any(r['gate']=='decode C3 4096' and r['status']=='PASS' for r in records) else 1)
