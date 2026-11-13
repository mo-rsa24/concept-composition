#!/bin/bash
# SynGen ported to SDXL (Table 1): the shared-device launcher for one card of a biggpu node.
# Plan: _archive/1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md, Plan 2.
#
#   STAGE=smoke  step size 0 on cat and dog (against r1_joint), step size 20 on two prompts with modifiers, contact sheet
#   STAGE=run    production, SHARD=i/n
#
#   timeout 30 ssh <node> 'GPU=<idx> STAGE=smoke REPO=<checkout> PY=<co3 or co3_bw python> nohup bash \
#       <checkout>/scripts/review_syngen_sdxl.sh > /datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/<name>.log \
#       2>&1 < /dev/null & disown'
#
# PY is co3 on the RTX 8000 and A6000 nodes (106 to 109) and co3_bw on the Blackwell nodes (110 to 112), whose cards co3
# has no kernels for. Every path is absolute; the script cd's into REPO itself.
set -uo pipefail
REPO="${REPO:-/home-mscluster/mmolefe/Playground/PhD/concept-composition}"
GPU="${GPU:?set GPU to the device index}"
STAGE="${STAGE:?set STAGE to smoke or run}"
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
export CUDA_VISIBLE_DEVICES="$GPU" POE_REPO="$REPO" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

echo "=== syngen-sdxl $STAGE ${SHARD:-}  node $(hostname -s)  device $GPU  pid $$  python $PY  $(date -Is)"
echo "=== repo $REPO at $(git -C "$REPO" rev-parse HEAD)  syngen $(git -C /datasets/mmolefe/repos/Linguistic-Binding-in-Diffusion-Models rev-parse --short HEAD)"
mkdir -p "$OUT/logs" "$OUT/syngen-sdxl_default/r1" || { echo "ABORT: cannot create $OUT" >&2; exit 2; }
PROBE="$OUT/.probe_$(hostname -s)_$$"
echo ok > "$PROBE" && rm -f "$PROBE" || { echo "ABORT: cannot write to $OUT from $(hostname -s)" >&2; exit 2; }
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
read -r USED UTIL TEMP < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL$TEMP" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted (util $UTIL, temp $TEMP)" >&2; exit 4;; esac
for pid in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -z "$owner" ] || [ "$owner" = "$USER" ] || { echo "ABORT: device $GPU runs pid $pid of $owner" >&2; exit 4; }
done
if [ "$USED" -ge 1024 ] && [ "${SHARE_OWN:-0}" != 1 ]; then echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; fi
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
echo "=== disk ${USE}%  device $GPU $(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader) ${USED} MiB ${UTIL}%"
cd "$REPO" || exit 7

case "$STAGE" in
  smoke) "$PY" 1-lora-correction/scripts/review_syngen_sdxl_cell.py --smoke; RC=$? ;;
  run)   "$PY" 1-lora-correction/scripts/review_syngen_sdxl_cell.py --shard "${SHARD:-0/1}"; RC=$?
         echo "=== $(ls "$OUT/syngen-sdxl_default/r1"/*/seed*/syngen-sdxl.json 2>/dev/null | wc -l) sidecars of 1104 on disk" ;;
  *) echo "ABORT: unknown STAGE $STAGE" >&2; exit 2 ;;
esac
echo "=== done $STAGE ${SHARD:-}  rc $RC  $(date -Is)"
exit $RC
