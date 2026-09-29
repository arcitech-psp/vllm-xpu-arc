#!/usr/bin/env bash
# Settings on the 12 GB system (Intel Arc B580): 8 sequences, 55% of the card for the FP8 and INT8 builds (4.55 GiB
# of weights). The BF16 build's weights alone are 7.87 GiB, so give it a larger share, e.g. UTIL=0.85 (we did not
# record the exact value used for the BF16 runs).
exec env GPU_INDEX="${GPU_INDEX:-0}" UTIL="${UTIL:-0.55}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}" \
  "$(dirname -- "$0")/serve-decision.sh" "$@"
