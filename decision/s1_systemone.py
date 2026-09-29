"""TypeSafe /v1/systemone front end for a single-token-label decision model served by vLLM (Intel XPU).

Wire format, request parsing and answer shapes are Mica's own (mica.typesafe_server.rows_from_request / answer),
so JevBench's `typesafe` adapter and other TypeSafe clients call it unchanged. Scoring is Mica's method: one prefill,
the answer restricted to the codebook label tokens, the raw label logits read back (vLLM --logprobs-mode
processed_logits + allowed_token_ids), divided by the fitted temperature, softmax. Prompt = mica.native.prompt_for().
Needs Mica's own `mica` package (github.com/akivet/Mica-v0.1-4B) on the path: set MICA_SRC to that clone.
usage: MICA_SRC=/path/to/Mica-v0.1-4B python3 s1_systemone.py --vllm http://127.0.0.1:8020 --model mintelica
         --hf <model dir> --calibration <model dir>/calibration.json --port 8012
"""
import argparse, json, math, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.environ.get('MICA_SRC', '/mica'))
from transformers import AutoTokenizer
from mica.native import prompt_for
from mica.codebook import OPTION_LABELS, NOUL_LABELS
from mica.typesafe_server import rows_from_request, answer

ap = argparse.ArgumentParser()
ap.add_argument('--vllm', default='http://127.0.0.1:8020'); ap.add_argument('--model', default='mintelica')
ap.add_argument('--hf', required=True); ap.add_argument('--calibration'); ap.add_argument('--port', type=int, default=8012)
ap.add_argument('--host', default='0.0.0.0'); ap.add_argument('--name', default='mintelica')
ap.add_argument('--max-tokens', type=int, default=8000)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(args.hf)
LABEL = [tok.encode(l, add_special_tokens=False)[0] for l in OPTION_LABELS]
NOUL = [tok.encode(l, add_special_tokens=False)[0] for l in NOUL_LABELS]   # No, Yes
cal = json.load(open(args.calibration)) if args.calibration else {}
TEMP = float(cal.get('temperature', 1.0))
POOL = ThreadPoolExecutor(8)


def label_logits(ids_prompt, ids):
    body = {'model': args.model, 'prompt': ids_prompt, 'max_tokens': 1, 'temperature': 0,
            'logprobs': len(ids), 'allowed_token_ids': ids, 'return_tokens_as_token_ids': True}
    req = urllib.request.Request(f'{args.vllm}/v1/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    top = json.load(urllib.request.urlopen(req, timeout=120))['choices'][0]['logprobs']['top_logprobs'][0]
    got = {int(k.split(':')[1]): v for k, v in top.items()}
    return [got.get(i, -1e9) for i in ids]


def probs(row):
    ids = NOUL if row['kind'] == 'noul' else LABEL[:len(row['candidates'])]
    p = tok(prompt_for(row, tok), add_special_tokens=False).input_ids
    if len(p) > args.max_tokens:
        raise ValueError(f"input is {len(p)} tokens, over the {args.max_tokens}-token contract")
    z = [v / TEMP for v in label_logits(p, ids)]; m = max(z); w = [math.exp(v - m) for v in z]; t = sum(w)
    return [x / t for x in w], len(p)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        self._send(200, {'status': 'ok', 'ready': True, 'model': args.name, 'temperature': TEMP})
    def do_POST(self):
        t0 = time.time()
        try:
            rows = rows_from_request(json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}'))
            res = list(POOL.map(probs, rows))
        except ValueError as e:
            return self._send(400, {'error': str(e)})
        except Exception as e:
            return self._send(500, {'error': str(e)[:300]})
        self._send(200, {'model': args.name, 'answers': {r['id']: answer(r, p) for r, (p, _) in zip(rows, res)},
                         'usage': {'input_tokens': sum(n for _, n in res), 'output_tokens': 0},
                         'latency_ms': round((time.time() - t0) * 1000, 1)})


print(json.dumps({'serving': f'http://{args.host}:{args.port}/v1/systemone', 'model': args.name, 'temperature': TEMP}), flush=True)
ThreadingHTTPServer((args.host, args.port), H).serve_forever()
