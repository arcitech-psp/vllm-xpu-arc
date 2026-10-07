#!/usr/bin/env bash
set -euo pipefail
WORK=/home/psp/fork-update
REPO=$WORK/vllm-xpu-arc
IMAGE=${VLLM_XPU_IMAGE:-vllm-xpu-arc:v030-20261006}
SERVER_CONTAINER=${CONTAINER_NAME:-fork-update-holo4-b70}
GATE_CONTAINER=${GATE_CONTAINER_NAME:-fork-update-holo4-gates}
[[ -f /home/psp/holo4/B70-FREE ]] || exit 75
mkdir -p "$WORK/evidence"
python3 "$REPO/scripts/b70-clients.py" > "$WORK/evidence/drm-clients-before.json"
docker inspect "$SERVER_CONTAINER" --format '{{.Image}} {{.State.Pid}} {{.State.StartedAt}}' \
  > "$WORK/evidence/server-identity.txt"
# CPU-only client container: the only inference target is our checked endpoint.
docker run --name "$GATE_CONTAINER" --network host --memory=2g \
  --label "arcitech.fork-update-run=${FORK_UPDATE_RUN_ID:-manual}" \
  -v /home/psp/holo4/chunks/c0:/models/holo4:ro \
  -v /home/psp/holo4/B70-FREE:/B70-FREE:ro \
  -v "$WORK/runtime:/templates:ro" -v "$WORK:/results" -v "$REPO:/repo:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint /opt/venv/bin/python "$IMAGE" \
  /repo/scripts/gate_holo4.py --output /results/evidence \
  > "$WORK/gates-runner.log" 2>&1
docker logs "$SERVER_CONTAINER" > "$WORK/serve-holo4-final.log" 2>&1
python3 "$REPO/scripts/b70-clients.py" > "$WORK/evidence/drm-clients-after.json"
