#!/usr/bin/env bash
# JevBench public tiers (easy / original / hard) against a TypeSafe /v1/systemone endpoint, unchanged `typesafe` adapter.
# usage: JEVBENCH_DIR=/path/to/jevbench bash run_jev.sh ENDPOINT LABEL      (Linux, or Git Bash on Windows)
# Writes RUNS_DIR/<LABEL>-{easy,original,hard}.jsonl and <LABEL>.jsonl, then prints the per-tier summary (analyze.py).
set -euo pipefail
EP=$1; LABEL=$2
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
JB=${JEVBENCH_DIR:?clone https://github.com/fstandhartinger/jevbench and set JEVBENCH_DIR}
PY=${PYTHON:-python3}
OUT=${RUNS_DIR:-$PWD/runs}; mkdir -p "$OUT"
if command -v cygpath >/dev/null 2>&1; then
  p() { cygpath -w "$1"; }
  # JevBench's ledger imports the POSIX-only fcntl; shims/fcntl.py is a no-op stand-in (runs are serial).
  export PYTHONPATH="$(p "$JB");$(p "$HERE/shims")"
else
  p() { echo "$1"; }
  export PYTHONPATH="$JB${PYTHONPATH:+:$PYTHONPATH}"
fi
export JEVBENCH_KEY=${JEVBENCH_KEY:-local}
for t in easy original hard; do
  "$PY" -m jevbench.cli run --tasks "$(p "$JB/datasets/public/$t.jsonl")" --adapter typesafe --endpoint "$EP" --model "$LABEL" \
    --key-env JEVBENCH_KEY --results "$(p "$OUT/$LABEL-$t.jsonl")" --ledger "$(p "$OUT/$LABEL-ledger.jsonl")" \
    --raw-dir "$(p "$OUT/raw-$LABEL")" --run-label "$LABEL" 2>&1 | tail -1
done
cat "$OUT/$LABEL-easy.jsonl" "$OUT/$LABEL-original.jsonl" "$OUT/$LABEL-hard.jsonl" > "$OUT/$LABEL.jsonl"
JEVBENCH_DIR="$(p "$JB")" "$PY" "$(p "$HERE/analyze.py")" "$(p "$OUT/$LABEL.jsonl")"
