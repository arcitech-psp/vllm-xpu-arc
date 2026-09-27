#!/bin/sh
set -eu

# One identical host-side measurement block. Run it three times back-to-back
# on the same container and compare medians from the saved block logs.
CONTAINER=${1:?container name required}
OUT=${2:?output directory required}
BENCH8000_IN_CONTAINER=${BENCH8000_IN_CONTAINER:-/tmp/bench8000.py}
TFAST_IN_CONTAINER=${TFAST_IN_CONTAINER:-/work/prewarm/tfast_bench.py}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
MODEL=${MODEL:-tiel-coder}
URL=${URL:-http://127.0.0.1:8000}
TFAST_REPS=${TFAST_REPS:-3}
mkdir -p "$OUT"

stamp=$(date +%Y%m%dT%H%M%S%z)
exec >"$OUT/block-$stamp.log" 2>&1
echo "block=$stamp container=$CONTAINER"
echo "date=$(date --iso-8601=seconds)"
echo '--- docker ---'
docker ps --no-trunc --format '{{.Names}} {{.Image}} {{.Status}} {{.Ports}}'
echo '--- memory ---'
free -g
echo '--- XPU power/temp ---'
for p in /sys/class/drm/card*/device/hwmon/hwmon*/power1_average /sys/class/drm/card*/device/hwmon/hwmon*/temp*_input; do
    [ -e "$p" ] && printf '%s=' "$p" && cat "$p"
done
echo '--- container config ---'
docker inspect "$CONTAINER" --format 'image={{.Config.Image}} cmd={{json .Config.Cmd}} env={{json .Config.Env}}'
echo '--- shape priming (excluded from measured rows) ---'
docker exec "$CONTAINER" "$PYTHON_BIN" "$BENCH8000_IN_CONTAINER" 1 >/dev/null
docker exec "$CONTAINER" "$PYTHON_BIN" "$BENCH8000_IN_CONTAINER" 4 >/dev/null
echo '--- bench8000 1 stream ---'
docker exec "$CONTAINER" "$PYTHON_BIN" "$BENCH8000_IN_CONTAINER" 1
echo '--- bench8000 4 streams ---'
docker exec "$CONTAINER" "$PYTHON_BIN" "$BENCH8000_IN_CONTAINER" 4
echo '--- tfast short and long cells ---'
docker exec "$CONTAINER" "$PYTHON_BIN" "$TFAST_IN_CONTAINER" \
    --url "$URL" --model "$MODEL" --tokenizer "${TOKENIZER_IN_CONTAINER:-/models/tiel-coder}" \
    --reps "$TFAST_REPS" --out "/tmp/tfinal-tfast-$stamp.json"
docker cp "$CONTAINER:/tmp/tfinal-tfast-$stamp.json" "$OUT/tfast-$stamp.json"
echo "saved=$OUT/tfast-$stamp.json"
