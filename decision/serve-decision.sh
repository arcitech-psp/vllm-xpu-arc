#!/usr/bin/env bash
# Serve a Mintelica build (sky7350's Mica-v0.1-4B for Intel Arc) with this repository's vLLM XPU image and the
# TypeSafe /v1/systemone front end (s1_systemone.py). vLLM listens inside the container on 127.0.0.1:8020;
# the front end is published on the host at http://localhost:$PORT/v1/systemone.
#
#   MODEL_DIR=/path/to/Mica-v0.1-4B-FP8-XPU MICA_SRC=/path/to/Mica-v0.1-4B ./decision/serve-decision.sh
#
# Tested presets: serve-b580.sh (12 GB Arc B580) and serve-b70.sh (32 GB Arc Pro B70).
set -euo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(dirname "$HERE")

IMAGE=${VLLM_XPU_IMAGE:-vllm-xpu-arc:local}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the downloaded model folder (it must contain calibration.json)}
MICA_SRC=${MICA_SRC:?set MICA_SRC to a clone of https://github.com/akivet/Mica-v0.1-4B (its prompt and codebook)}
NAME=${NAME:-mintelica}
PORT=${PORT:-8012}
GPU_INDEX=${GPU_INDEX:-0}          # ZE_AFFINITY_MASK: index of the Arc card that runs the model
UTIL=${UTIL:-0.55}                 # --gpu-memory-utilization
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_TOKENS=${MAX_TOKENS:-8000}     # longer inputs are refused with HTTP 400, never truncated
W8A8=${W8A8:-0}                    # FP8 build only: 1 = FP8 math (XPUW8A8FP8LinearKernel); 0 = FP8 weights, BF16 math
EXTRA=${EXTRA:-}
RENDER_GROUP=${RENDER_GROUP:-$(stat -c '%g' /dev/dri/renderD* 2>/dev/null | head -1 || true)}
RENDER_GROUP=${RENDER_GROUP:-render}
if [ "$W8A8" = 1 ]; then EXTRA="--linear-backend xpu $EXTRA"; fi
[ -f "$MODEL_DIR/calibration.json" ] || { echo "missing $MODEL_DIR/calibration.json" >&2; exit 1; }

hier=()
if [ -n "${ZE_FLAT_DEVICE_HIERARCHY:-}" ]; then hier=(-e "ZE_FLAT_DEVICE_HIERARCHY=$ZE_FLAT_DEVICE_HIERARCHY"); fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
# Why these flags (README, "Decision models on Arc"):
#  * --device /dev/dri --privileged --group-add <render gid>: the tested device setup.
#  * ZE_AFFINITY_MASK: pin the model to one Arc card.
#  * SYCL_CACHE_PERSISTENT=0: the persistent SYCL device-code cache segfaulted on the first request.
#  * The patched scheduler imports adaptive_mtp at startup even with adaptive MTP off. Setting PYTHONPATH
#    replaces the image's own entry, so the module is mounted and kept on the path explicitly.
docker run -d --name "$NAME" -p "$PORT:8012" \
  --device /dev/dri --privileged --group-add "$RENDER_GROUP" --ipc host \
  -e ZE_AFFINITY_MASK="$GPU_INDEX" ${hier[@]+"${hier[@]}"} \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e SYCL_CACHE_PERSISTENT=0 -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  -e VLLM_XPU_ADAPTIVE_MTP=0 -e VLLM_XPU_ADAPTIVE_GRAPHS=0 -e VLLM_XPU_MTP_POLICY=none -e ADAPTIVE_MTP_TELEMETRY=/dev/null \
  -e PYTHONPATH=/work/adaptive_mtp:/opt/vllm-xpu-arc/adaptive:/opt/vllm-xpu-arc/mxfp4 -e MICA_SRC=/mica \
  -v "$MODEL_DIR:/model:ro" -v "$MICA_SRC:/mica:ro" -v "$HERE:/decision:ro" \
  -v "$REPO/adaptive/adaptive_mtp.py:/work/adaptive_mtp/adaptive_mtp.py:ro" \
  --entrypoint bash "$IMAGE" -c "
  vllm serve /model --served-model-name $NAME --port 8020 --host 127.0.0.1 --dtype bfloat16 \
    --max-model-len 8192 --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $UTIL --enable-prefix-caching \
    --logprobs-mode processed_logits --max-logprobs 256 --enforce-eager $EXTRA > /tmp/vllm.log 2>&1 &
  until curl -sf http://127.0.0.1:8020/v1/models >/dev/null; do sleep 3; done
  exec python3 /decision/s1_systemone.py --vllm http://127.0.0.1:8020 --model $NAME --hf /model \
    --calibration /model/calibration.json --port 8012 --max-tokens $MAX_TOKENS --name $NAME"
echo "started $NAME -> http://localhost:$PORT/v1/systemone  (vLLM log: docker exec $NAME cat /tmp/vllm.log)"
