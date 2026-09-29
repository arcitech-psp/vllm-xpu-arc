"""Mean / min / max per JevBench tier over N runs of run_jev.sh (used by repeat_jev.sh).
usage: python summarize_runs.py LABEL N RUNS_DIR   ->  RUNS_DIR/<LABEL>-summary.json
"""
import json, os, statistics as st, sys

label, n, runs_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
runs = []
for k in range(1, n + 1):
    s = open(os.path.join(runs_dir, f'jev-{label}-r{k}.log')).read()
    runs.append(json.loads(s[s.index('{'):])['tiers'])
out = {}
for t in runs[0]:
    acc = [r[t]['accuracy'] for r in runs]
    p50 = [r[t]['p50_ms'] for r in runs if r[t]['p50_ms']]
    ece = [r[t]['ece'] for r in runs if r[t]['ece'] is not None]
    out[t] = {'acc_mean': round(st.mean(acc), 2), 'acc_min': min(acc), 'acc_max': max(acc),
              'ece_mean': round(st.mean(ece), 3) if ece else None,
              'p50_ms_mean': round(st.mean(p50), 1) if p50 else None,
              'p50_ms_min': min(p50) if p50 else None, 'p50_ms_max': max(p50) if p50 else None,
              'failed_max': max(r[t]['n_failed_or_refused'] for r in runs)}
json.dump({'label': label, 'runs': n, 'tiers': out}, open(os.path.join(runs_dir, f'{label}-summary.json'), 'w'), indent=1)
print(label, json.dumps(out))
