#!/bin/sh
set -eu

# Start vLLM, wait for health, then run one discardable graph/MTP warmup.
VLLM_BIN=${VLLM_BIN:-/opt/venv/bin/vllm}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
SERVER_PORT=${SERVER_PORT:-8000}
HEALTH_URL=${HEALTH_URL:-http://127.0.0.1:${SERVER_PORT}/health}
PREWARM_ENABLED=${TIEL_K3_PREWARM:-1}
PREWARM_SCRIPT=${PREWARM_SCRIPT:-/work/prewarm/prewarm_shortreply.py}
PREWARM_TOKENIZER=${TIEL_PREWARM_TOKENIZER:-}
PREWARM_MODEL=${TIEL_PREWARM_MODEL:-tiel-coder}
PREWARM_URL=${TIEL_PREWARM_URL:-http://127.0.0.1:${SERVER_PORT}}
PREWARM_PROMPT_TOKENS=${TIEL_PREWARM_PROMPT_TOKENS:-512}
PREWARM_MAX_TOKENS=${TIEL_PREWARM_MAX_TOKENS:-32}

"$VLLM_BIN" "$@" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' TERM INT

i=0
while ! curl -fsS "$HEALTH_URL" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid" || true
        exit 1
    fi
    i=$((i + 1))
    if [ "$i" -ge "${HEALTH_TIMEOUT_POLLS:-180}" ]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" || true
        echo "timed out waiting for vLLM health at $HEALTH_URL" >&2
        exit 1
    fi
    sleep "${HEALTH_POLL_SECONDS:-2}"
done

if [ "$PREWARM_ENABLED" = "1" ]; then
    : "${PREWARM_TOKENIZER:?set TIEL_PREWARM_TOKENIZER to the mounted model/tokenizer path}"
    TIEL_K3_PREWARM=1 PYTHONPATH=${PYTHONPATH:-} \
        "$PYTHON_BIN" "$PREWARM_SCRIPT" \
        --url "$PREWARM_URL" \
        --model "$PREWARM_MODEL" \
        --tokenizer "$PREWARM_TOKENIZER" \
        --prompt-tokens "$PREWARM_PROMPT_TOKENS" \
        --max-tokens "$PREWARM_MAX_TOKENS"
fi

wait "$server_pid"
