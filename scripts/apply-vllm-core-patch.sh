#!/usr/bin/env bash
set -euo pipefail

# Apply the patch to a checkout of vLLM at the pinned upstream commit.
# Run from the repository root: scripts/apply-vllm-core-patch.sh /path/to/vllm
root=${1:?usage: $0 /path/to/vllm-checkout}
test -d "$root/vllm"
test "$(git -C "$root" rev-parse HEAD)" = \
  "ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9"
git -C "$root" apply --check \
  "$PWD/patches/0001-vllm-ac7509e2b-xpu-extras.patch"
git -C "$root" apply \
  "$PWD/patches/0001-vllm-ac7509e2b-xpu-extras.patch"
install -m 0644 "$PWD/patches/vllm_xpu_draft_lmhead_int4.py" \
  "$root/vllm/model_executor/models/vllm_xpu_draft_lmhead_int4.py"
install -m 0644 "$PWD/patches/vllm_xpu_draft_mtp_int4.py" \
  "$root/vllm/model_executor/models/vllm_xpu_draft_mtp_int4.py"
