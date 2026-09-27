#!/usr/bin/env bash
set -euo pipefail

# Fixed-K3 fast route. Supply the model and optional runtime overlay paths;
# nothing here depends on a developer machine layout.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IMAGE=${VLLM_XPU_IMAGE:-vllm-xpu-arc:local}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the host model directory}
MODEL_MOUNT=${MODEL_MOUNT:-/models/tiel-coder}
MODEL_NAME=${MODEL_NAME:-tiel-coder}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$MODEL_MOUNT/chat_template.jinja}
TOKENIZER_DIR=${TOKENIZER_DIR:-$MODEL_MOUNT}
CONTAINER_NAME=${CONTAINER_NAME:-tiel-coder-fast}
PORT=${PORT:-8000}
ADAPTIVE_DIR=${ADAPTIVE_DIR:-}
ADAPTIVE_MTP_SCRIPT=${ADAPTIVE_MTP_SCRIPT:-}
SCHEDULER_PATCH=${SCHEDULER_PATCH:-}
BENCH8000_HOST=${BENCH8000_HOST:-}

DRAFT_LMHEAD_HOST=${DRAFT_LMHEAD_HOST:-$SCRIPT_DIR/../patches/vllm_xpu_draft_lmhead_int4.py}
DRAFT_MTP_HOST=${DRAFT_MTP_HOST:-$SCRIPT_DIR/../patches/vllm_xpu_draft_mtp_int4.py}
PREWARM_HOST=${PREWARM_HOST:-$SCRIPT_DIR/prewarm_shortreply.py}
TFAST_HOST=${TFAST_HOST:-$SCRIPT_DIR/../bench/tfast_bench.py}
ENTRYPOINT_HOST=${ENTRYPOINT_HOST:-$SCRIPT_DIR/tiel-mtp4-entrypoint.sh}

for required in "$MODEL_DIR" "$DRAFT_LMHEAD_HOST" "$DRAFT_MTP_HOST" "$PREWARM_HOST" "$TFAST_HOST" "$ENTRYPOINT_HOST"; do
    [ -e "$required" ] || { echo "missing required path: $required" >&2; exit 1; }
done

docker_args=(
    run --rm --name "$CONTAINER_NAME" -p "${PORT}:8000"
    --entrypoint /work/tiel-mtp4-adaptive/entrypoint.sh
    --device /dev/dri --ipc=host --shm-size 4g
    -v "$MODEL_DIR:$MODEL_MOUNT:ro"
    -v "$ENTRYPOINT_HOST:/work/tiel-mtp4-adaptive/entrypoint.sh:ro"
    -v "$DRAFT_LMHEAD_HOST:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/b70_draft_lmhead_int4.py:ro"
    -v "$DRAFT_MTP_HOST:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/b70_draft_mtp_int4.py:ro"
    -v "$PREWARM_HOST:/work/prewarm/prewarm_shortreply.py:ro"
    -v "$TFAST_HOST:/work/prewarm/tfast_bench.py:ro"
)

if [ -n "$ADAPTIVE_DIR" ]; then
    docker_args+=( -v "$ADAPTIVE_DIR:/work/adaptive:ro" )
fi
if [ -n "$ADAPTIVE_MTP_SCRIPT" ]; then
    docker_args+=( -v "$ADAPTIVE_MTP_SCRIPT:/work/tiel-mtp4-adaptive/adaptive_mtp.py:ro" )
fi
if [ -n "$SCHEDULER_PATCH" ]; then
    docker_args+=( -v "$SCHEDULER_PATCH:/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro" )
fi
if [ -n "$BENCH8000_HOST" ]; then
    docker_args+=( -v "$BENCH8000_HOST:/tmp/bench8000.py:ro" )
fi

exec docker "${docker_args[@]}" \
    -e VLLM_TARGET_DEVICE=xpu \
    -e PYTHONPATH=/work/tiel-mtp4-adaptive:/work/prewarm:/work/adaptive \
    -e B70_MTP_BF16_DRAFT=1 \
    -e B70_DRAFT_LMHEAD_INT4=1 \
    -e B70_DRAFT_MTP_INT4=1 \
    -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}" \
    -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
    -e SYCL_CACHE_PERSISTENT=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e VLLM_GGUF_USE_CUDA=0 \
    -e VLLM_GGUF_DENSE_FORMAT=fp8 \
    -e SERVER_PORT=8000 \
    -e TIEL_K3_PREWARM=1 \
    -e TIEL_PREWARM_TOKENIZER="$MODEL_MOUNT" \
    -e TIEL_PREWARM_MODEL="$MODEL_NAME" \
    -e TIEL_PREWARM_URL=http://127.0.0.1:8000 \
    "$IMAGE" serve "$MODEL_MOUNT" \
    --chat-template "$CHAT_TEMPLATE" \
    --served-model-name "$MODEL_NAME" \
    --dtype bfloat16 --kv-cache-dtype fp8 \
    --max-model-len "${MAX_MODEL_LEN:-131072}" \
    --max-num-seqs "${MAX_NUM_SEQS:-4}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}" \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --mm-processor-kwargs '{"max_pixels":4194304}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 --host 0.0.0.0 --port 8000 \
    --enable-prompt-tokens-details
