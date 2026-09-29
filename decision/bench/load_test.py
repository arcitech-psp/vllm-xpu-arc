"""Serving load test for a TypeSafe /v1/systemone endpoint: real JevBench request bodies (all public tiers, the
exact requests JevBench sent), replayed by C concurrent clients for a fixed duration each.
Reports per concurrency: decisions/s (requests and questions), p50/p95/p99 latency, errors.
usage: python load_test.py ENDPOINT LABEL --raw runs/raw-<label> [--conc 1,4,8,16] [--seconds 60] [--out runs]
  --raw is the raw-response directory of an earlier run_jev.sh run: its JSON files hold the exact JevBench requests.
"""
import argparse, glob, json, os, random, statistics as st, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('endpoint'); ap.add_argument('label')
ap.add_argument('--conc', default='1,4,8,16'); ap.add_argument('--seconds', type=float, default=60)
ap.add_argument('--raw', required=True); ap.add_argument('--out', default='runs')
args = ap.parse_args()
bodies = []
for f in sorted(glob.glob(os.path.join(args.raw, '*.json'))):
    r = json.load(open(f, encoding='utf-8')).get('request')
    if r: r.pop('model', None); bodies.append(json.dumps(r).encode())
random.seed(0)
print(f'{len(bodies)} real JevBench requests', flush=True)


def call(b):
    t = time.time()
    req = urllib.request.Request(f'{args.endpoint}/v1/systemone', data=b, headers={'Content-Type': 'application/json'})
    j = json.load(urllib.request.urlopen(req, timeout=300))
    return time.time() - t, len(j.get('answers', {}))


out = {'label': args.label, 'endpoint_note': 'LAN client', 'levels': []}
for c in [int(x) for x in args.conc.split(',')]:
    lat, qs, errs, stop = [], [0], [0], time.time() + args.seconds
    lock = threading.Lock()
    def worker(seed):
        rnd = random.Random(seed)
        while time.time() < stop:
            try:
                dt, nq = call(rnd.choice(bodies))
                with lock: lat.append(dt); qs[0] += nq
            except Exception:
                with lock: errs[0] += 1
    call(bodies[0])   # warm
    t0 = time.time(); th = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
    [x.start() for x in th]; [x.join() for x in th]; el = time.time() - t0
    lat.sort(); q = lambda p: round(1000 * lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None
    row = {'concurrency': c, 'requests': len(lat), 'req_per_s': round(len(lat) / el, 2), 'questions_per_s': round(qs[0] / el, 2),
           'p50_ms': q(0.5), 'p95_ms': q(0.95), 'p99_ms': q(0.99), 'errors': errs[0]}
    out['levels'].append(row); print(json.dumps(row), flush=True)
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, f'{args.label}-load.json'), 'w'), indent=1)
