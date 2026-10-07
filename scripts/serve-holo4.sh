#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK_DIR=${WORK_DIR:-/home/psp/fork-update}
FREE_MARKER=${B70_FREE_MARKER:-/home/psp/holo4/B70-FREE}
MODEL_DIR=${MODEL_DIR:-/home/psp/holo4/chunks/c0}
ADAPTER_DIR=${ADAPTER_DIR:-/home/psp/swift/jev/adapter_vllm}
IMAGE=${VLLM_XPU_IMAGE:-vllm-xpu-arc:v030-20261006}
CONTAINER_NAME=${CONTAINER_NAME:-fork-update-holo4-b70}
PORT=${PORT:-8000}
INC_BACKEND=${VLLM_XPU_INC_WNA16_BACKEND:-w4a16}  # auto picks ARK, whose kernel traps (invalid opcode) on the B70; w4a16 is the measured, served path (2026-10-07)
INT4_COMPUTE_DTYPE=${VLLM_XPU_INT4_COMPUTE_DTYPE:-native}
case "$INC_BACKEND" in auto|ark|w4a16|w4a8) ;; *) echo 'Invalid INC backend.' >&2; exit 64 ;; esac
case "$INT4_COMPUTE_DTYPE" in native|float16) ;; *) echo 'Invalid INT4 compute dtype.' >&2; exit 64 ;; esac
CACHE_DIR=$WORK_DIR/cache/${INC_BACKEND}-${INT4_COMPUTE_DTYPE}
TEMPLATE_DIR=$WORK_DIR/runtime
[[ -f "$FREE_MARKER" ]] || { echo 'B70-FREE absent: serving is gated.' >&2; exit 75; }
[[ -r "$MODEL_DIR/config.json" ]] && \
  [[ -r "$MODEL_DIR/model.safetensors.index.json" || -r "$MODEL_DIR/model.safetensors" ]] || {
  echo 'Exported model configuration and safetensors weights are required.' >&2; exit 75;
}
[[ -r "$ADAPTER_DIR/adapter_config.json" ]] || { echo 'JEV adapter missing.' >&2; exit 75; }
mkdir -p "$TEMPLATE_DIR" "$CACHE_DIR"
[[ "${SKIP_EXPORT_CHECK:-0}" = 1 ]] || python3 "$SCRIPT_DIR/check-holo4-export.py" "$MODEL_DIR" \
  > "$WORK_DIR/export-preflight.json"
python3 "$SCRIPT_DIR/prepare-holo4-template.py" --model "$MODEL_DIR" \
  --output "$TEMPLATE_DIR/chat_template_fast.jinja"
RG=$(stat -c '%g' /dev/dri/renderD* | sort -u | head -1)

# All mounts of model/adapter data are read-only. No clocks or services change.
# The preflight checks device identity and the original memory reservation before
# starting vLLM. A competing allocation is a failure, never a reason to hide it.
exec docker run --name "$CONTAINER_NAME" --device /dev/dri --group-add "$RG" \
  --label "arcitech.fork-update-run=${FORK_UPDATE_RUN_ID:-manual}" \
  --ipc=host --shm-size=4g --memory=18g --memory-swap=18g \
  -p "${PORT}:8000" \
  -v "$MODEL_DIR:/models/holo4:ro" -v "$ADAPTER_DIR:/adapters/jev:ro" \
  -v "$TEMPLATE_DIR:/templates:ro" -v "$CACHE_DIR:/cache" \
  -e ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-1}" -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e VLLM_TARGET_DEVICE=xpu -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}" \
  -e VLLM_XPU_MTP_BF16_DRAFT=1 -e VLLM_XPU_DRAFT_LMHEAD_INT4=${DRAFT_LMHEAD_INT4:-0} \
  -e VLLM_XPU_DRAFT_MTP_INT4=${DRAFT_MTP_INT4:-0} -e VLLM_XPU_INC_WNA16_BACKEND="$INC_BACKEND" \
  -e VLLM_XPU_INT4_COMPUTE_DTYPE="$INT4_COMPUTE_DTYPE" \
  -e SYCL_CACHE_PERSISTENT=0 -e VLLM_XPU_USE_SAMPLER_KERNEL=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint /opt/venv/bin/python "$IMAGE" \
  /opt/vllm-xpu-arc/scripts/holo4-entrypoint.py \
  /models/holo4 --served-model-name holo4-27b \
  --chat-template /templates/chat_template_fast.jinja \
  --dtype bfloat16 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype "${MAMBA_SSM_CACHE_DTYPE:-float16}" \
  --max-model-len 131072 --enable-prefix-caching \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}" \
  --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${MTP_TOKENS:-3}}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision=/adapters/jev \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --host 0.0.0.0 --port 8000 "$@"
