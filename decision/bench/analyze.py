"""Per-tier JevBench accuracy + latency for one run, optional per-item comparison with a reference run.
  JEVBENCH_DIR=/path/to/jevbench python analyze.py <run.jsonl> [reference.jsonl]
Uses JevBench's own summarize() on each tier, so accuracy is exactly the benchmark's rule.
"""
import json, sys, os, statistics
JB = os.environ.get('JEVBENCH_DIR') or sys.exit('set JEVBENCH_DIR to a clone of github.com/fstandhartinger/jevbench')
sys.path[:0] = [JB] + ([os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shims')] if os.name == 'nt' else [])
from jevbench.tasks import load_jsonl
from jevbench.summarize import summarize

args = [a for a in sys.argv[1:] if not a.startswith('--')]
run = [json.loads(l) for l in open(args[0], encoding='utf-8') if l.strip()]
ref = {r['task_id']: r for r in (json.loads(l) for l in open(args[1], encoding='utf-8') if l.strip())} if len(args) > 1 else None
by = {r['task_id']: r for r in run}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


out = {'run': os.path.basename(args[0]), 'tiers': {}}
all_lat = []
for tier in ('easy', 'original', 'hard'):
    tasks = load_jsonl(os.path.join(JB, 'datasets', 'public', f'{tier}.jsonl'))
    recs = [by[t.id] for t in tasks if t.id in by]
    s = summarize(tasks, recs, headline_only=True)
    lat = [r['latency_s'] * 1000 for r in recs if r['ok']]
    all_lat += lat
    row = {'n': len(tasks), 'n_correct': s['n_correct'], 'accuracy': round(100 * s['accuracy'], 1),
           'n_failed_or_refused': sum(1 for r in recs if not r['ok']), 'ece': round(s['ece']['ece'],3) if s['ece'] else None,
           'p50_ms': round(statistics.median(lat), 1) if lat else None, 'p90_ms': round(pct(lat, .9), 1) if lat else None}
    if ref:
        both = [t.id for t in tasks if t.id in ref]
        row['ref_accuracy'] = round(100 * sum(bool(ref[i]['correct']) for i in both) / len(both), 1)
        row['same_prediction_as_ref'] = sum(by[i].get('predicted') == ref[i].get('predicted') for i in both)
        row['max_abs_prob_diff_vs_ref'] = round(max((abs(by[i]['probs'].get(k, 0) - v) for i in both
                                                    if by[i].get('probs') and ref[i].get('probs')
                                                    for k, v in ref[i]['probs'].items()), default=0), 4)
        row['flips_vs_ref'] = [i for i in both if by[i].get('predicted') != ref[i].get('predicted')]
    out['tiers'][tier] = row
tot_n = sum(v['n'] for v in out['tiers'].values())
out['all'] = {'n': tot_n, 'n_correct': sum(v['n_correct'] for v in out['tiers'].values()),
              'p50_ms': round(statistics.median(all_lat), 1), 'p90_ms': round(pct(all_lat, .9), 1),
              'p95_ms': round(pct(all_lat, .95), 1), 'n_latency': len(all_lat)}
if ref:
    rl = [r['latency_s'] * 1000 for r in ref.values() if r['ok']]
    out['ref_latency'] = {'p50_ms': round(statistics.median(rl), 1), 'p90_ms': round(pct(rl, .9), 1), 'p95_ms': round(pct(rl, .95), 1)}
print(json.dumps(out, indent=1))
