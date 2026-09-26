#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-vllm-xpu-arc:local}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to a mounted model directory}
MODEL_NAME=${MODEL_NAME:-vllm-xpu-model}

exec docker run --rm --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/dri/render* | sort -u | head -1)" \
  -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -p "${PORT:-8000}:8000" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  "$IMAGE" serve /model \
  --served-model-name "$MODEL_NAME" \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8000
