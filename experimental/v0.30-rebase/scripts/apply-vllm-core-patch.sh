#!/usr/bin/env bash
set -euo pipefail

# Apply the patch to a checkout of vLLM v0.30.0 at the pinned upstream commit.
# Run from the repository root: scripts/apply-vllm-core-patch.sh /path/to/vllm
root=${1:?usage: $0 /path/to/vllm-checkout}
test -d "$root/vllm"
test "$(git -C "$root" rev-parse HEAD)" = \
  "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
git -C "$root" apply --check \
  "$PWD/experimental/v0.30-rebase/patches/0001-vllm-v0.30.0-ced6857-xpu-extras.patch"
git -C "$root" apply \
  "$PWD/experimental/v0.30-rebase/patches/0001-vllm-v0.30.0-ced6857-xpu-extras.patch"
install -m 0644 "$PWD/patches/vllm_xpu_draft_lmhead_int4.py" \
  "$root/vllm/model_executor/models/vllm_xpu_draft_lmhead_int4.py"
install -m 0644 "$PWD/patches/vllm_xpu_draft_mtp_int4.py" \
  "$root/vllm/model_executor/models/vllm_xpu_draft_mtp_int4.py"
