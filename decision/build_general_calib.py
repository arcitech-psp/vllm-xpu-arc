"""General-purpose calibration prompts for quantizing a decision model (GPTQ / sensitivity scans).

Rows are rendered in Mica's row format ({state, question, kind, candidates, answer}) so mica.native.prompt_for()
turns them into exactly the prompts the model serves. Sources are public, permissively licensed datasets read through
the Hugging Face datasets-server API (no JevBench items, so the benchmark stays clean):
  google/boolq (yes/no), tau/commonsense_qa (5-way), allenai/ai2_arc ARC-Easy + ARC-Challenge (4-way),
  fancyzhx/ag_news (topic), mteb/banking77 (intent; 1 in 5 rows drops the gold label -> "none of these").
usage: python build_general_calib.py OUT.jsonl [--per-source 140] [--seed 0]
"""
import argparse, json, random, urllib.parse, urllib.request

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--per-source', type=int, default=140)
ap.add_argument('--seed', type=int, default=0); args = ap.parse_args()
rnd = random.Random(args.seed)
NONE = {'id': 'none', 'text': 'none of these fits'}


def rows(dataset, config, split, n):
    out, off = [], 0
    while len(out) < n:
        q = urllib.parse.urlencode({'dataset': dataset, 'config': config, 'split': split, 'offset': off, 'length': 100})
        with urllib.request.urlopen(f'https://datasets-server.huggingface.co/rows?{q}', timeout=60) as r:
            got = [x['row'] for x in json.load(r)['rows']]
        if not got: break
        out += got; off += 100
    return out[:n]


def choice(state, question, texts, gold):
    order = list(range(len(texts))); rnd.shuffle(order)
    cands = [{'id': f'o{i + 1}', 'text': str(texts[j])} for i, j in enumerate(order)]
    return {'state': state, 'question': question, 'kind': 'choice', 'candidates': cands, 'answer': order.index(gold)}


N = args.per_source; out = []
for r in rows('google/boolq', 'default', 'train', N):
    out.append({'state': r['passage'][:1500], 'question': r['question'].rstrip('?') + '?', 'kind': 'noul',
                'candidates': [{'id': 'false', 'text': 'no'}, {'id': 'true', 'text': 'yes'}], 'answer': int(bool(r['answer']))})
for r in rows('tau/commonsense_qa', 'default', 'train', N):
    labels = r['choices']['label']
    out.append(choice('', r['question'], r['choices']['text'], labels.index(r['answerKey'])))
for cfg in ('ARC-Easy', 'ARC-Challenge'):
    for r in rows('allenai/ai2_arc', cfg, 'train', N // 2):
        labels = r['choices']['label']
        if r['answerKey'] in labels:
            out.append(choice('', r['question'], r['choices']['text'], labels.index(r['answerKey'])))
TOPICS = ['World news', 'Sports', 'Business', 'Science and technology']
for r in rows('fancyzhx/ag_news', 'default', 'train', N):
    out.append(choice(r['text'][:1200], 'Which topic is this article about?', TOPICS, int(r['label'])))
b77 = rows('mteb/banking77', 'default', 'train', 1500)          # label_text included; collect the label names
names = {int(r['label']): r['label_text'] for r in b77}
labels = sorted(names)
for r in b77[:N]:
    gold, drop = int(r['label']), rnd.random() < 0.2
    others = [l for l in labels if l != gold]; rnd.shuffle(others)
    idx = others[:4] + ([] if drop else [gold]); rnd.shuffle(idx)
    cands = [{'id': f'o{i + 1}', 'text': names[j].replace('_', ' ')} for i, j in enumerate(idx)] + [NONE]
    ans = len(cands) - 1 if drop else idx.index(gold)
    out.append({'state': r['text'], 'question': 'Which request type is this customer message?', 'kind': 'choice', 'candidates': cands, 'answer': ans})
rnd.shuffle(out)
with open(args.out, 'w', encoding='utf-8') as f:
    for x in out: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(json.dumps({'rows': len(out), 'noul': sum(x['kind'] == 'noul' for x in out), 'choice': sum(x['kind'] == 'choice' for x in out)}))
