#!/usr/bin/env bash
# Shared-device launch with the guards environment/hpc/execution-protocol.md requires.
# Usage (on the target node, every path absolute):
#   GPU=<idx> PY=<python> nohup bash launch_gpu.sh <script.py> [args...] > <log> 2>&1 < /dev/null &
set -uo pipefail
OUT=/datasets/mmolefe/poe_repair_min/outputs/classifier_guidance_photos
: "${GPU:?set GPU}"; : "${PY:?set PY}"
echo "launch $(date -Is) node $(hostname) GPU $GPU PID $$ PY $PY"
echo "cmd: $*"

[ -d "$OUT" ] || { echo "ABORT: $OUT missing"; exit 2; }
use=$(df --output=pcent "$OUT" | tail -1 | tr -dc 0-9)
[ -n "$use" ] && [ "$use" -lt 90 ] || { echo "ABORT: disk at '${use}'% on $OUT"; exit 2; }
[ -x "$PY" ] || { echo "ABORT: no python at $PY"; exit 2; }

row=$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits) \
  || { echo "ABORT: nvidia-smi failed on GPU $GPU"; exit 2; }
echo "device $GPU: $row (MiB, %, C)"
case "$row" in *N/A*|*reset*) echo "ABORT: GPU $GPU faulted"; exit 2;; esac
mem=$(echo "$row" | cut -d, -f1 | tr -dc 0-9)
[ "$mem" -lt 1024 ] || { echo "ABORT: GPU $GPU already holds ${mem} MiB"; exit 2; }

export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1
exec "$PY" -I -u "$@"
