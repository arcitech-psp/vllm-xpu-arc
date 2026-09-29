#!/usr/bin/env bash
# Tested settings on the 32 GB system (Intel Arc Pro B70): 16 sequences, 50% of the card, composite device hierarchy.
# Our B70 system also holds a second, idle Arc GPU, so the tested run pinned the B70 with GPU_INDEX=1. Set yours.
exec env GPU_INDEX="${GPU_INDEX:-0}" UTIL="${UTIL:-0.5}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" \
  ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-COMPOSITE}" \
  "$(dirname -- "$0")/serve-decision.sh" "$@"
