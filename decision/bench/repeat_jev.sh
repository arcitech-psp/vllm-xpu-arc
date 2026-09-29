#!/usr/bin/env bash
# JevBench N times against one endpoint, then mean / min / max per tier -> RUNS_DIR/<LABEL>-summary.json
# usage: JEVBENCH_DIR=/path/to/jevbench bash repeat_jev.sh ENDPOINT LABEL [N=3]
set -euo pipefail
EP=$1; LABEL=$2; N=${3:-3}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export RUNS_DIR=${RUNS_DIR:-$PWD/runs}; mkdir -p "$RUNS_DIR"
for k in $(seq 1 "$N"); do
  bash "$HERE/run_jev.sh" "$EP" "$LABEL-r$k" > "$RUNS_DIR/jev-$LABEL-r$k.log" 2>&1
done
RD=$RUNS_DIR
if command -v cygpath >/dev/null 2>&1; then RD=$(cygpath -w "$RUNS_DIR"); fi
"${PYTHON:-python3}" "$HERE/summarize_runs.py" "$LABEL" "$N" "$RD"
