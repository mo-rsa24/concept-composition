#!/bin/bash
# Compute job 10 on SD 1.5 (Table 4, Figure 9): the shared-device launcher for one card of a biggpu node.
# Plan: 1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md, Plan 1.
#
#   STAGE=prep   latents for seeds 9-12 and the token-index file (co3, CPU work)
#   STAGE=smoke  the three smoke cells per method, the PNDM and loss-off variants, then the checks and contact sheet
#   STAGE=check  the checks alone
#   STAGE=run    production: METHOD=joint+poe|aae|dnb|syngen (a + joins several, run in turn), SHARD=i/n
#
#   timeout 30 ssh mscluster107 'GPU=0 STAGE=smoke REPO=<checkout> nohup bash <checkout>/scripts/review_sd15.sh \
#       > /datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15/logs/<name>.log 2>&1 < /dev/null & disown'
#
# Every path is absolute and the script cd's into REPO itself, so the launch line needs no working directory.
# SHARE_OWN=1 lets a second process of this user's share a card already holding this user's work (never anyone else's).
set -uo pipefail
REPO="${REPO:-/home-mscluster/mmolefe/Playground/PhD/poe_repair_min}"
GPU="${GPU:?set GPU to the device index}"
STAGE="${STAGE:?set STAGE to prep, smoke, check or run}"
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review/sd15
CO3=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
AAE_PY=/datasets/mmolefe/envs/sd15-aae/bin/python
SYN_PY=/datasets/mmolefe/envs/syngen/bin/python
export CUDA_VISIBLE_DEVICES="$GPU" POE_REPO="$REPO" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
py_for() { case "$1" in joint|poe) echo "$CO3";; aae|dnb) echo "$AAE_PY";; syngen) echo "$SYN_PY";; *) echo "";; esac; }

echo "=== sd15 $STAGE ${METHOD:-} ${SHARD:-}  node $(hostname -s)  device $GPU  pid $$  job ${SLURM_JOB_ID:-none}  $(date -Is)"
echo "=== repo $REPO at $(git -C "$REPO" rev-parse HEAD)  syngen $(git -C /datasets/mmolefe/repos/Linguistic-Binding-in-Diffusion-Models rev-parse --short HEAD)  dnb $(git -C /datasets/mmolefe/repos/Divide-and-Bind rev-parse --short HEAD)"

# 1. The output folder exists and takes a write from this node (mscluster72 and 73 once could not see /datasets).
mkdir -p "$OUT/logs" "$OUT/smoke" || { echo "ABORT: cannot create $OUT" >&2; exit 2; }
PROBE="$OUT/.probe_$(hostname -s)_$$"
echo ok > "$PROBE" && rm -f "$PROBE" || { echo "ABORT: cannot write to $OUT from $(hostname -s)" >&2; exit 2; }
# 2. The disk guard reads the filesystem the tiles land on.
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
# 3. The card: numeric readings (a faulted card reads [N/A]), no other user's process, and empty unless SHARE_OWN=1.
read -r USED UTIL TEMP < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL$TEMP" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted (util $UTIL, temp $TEMP)" >&2; exit 4;; esac
for pid in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -z "$owner" ] || [ "$owner" = "$USER" ] || { echo "ABORT: device $GPU runs pid $pid of $owner" >&2; exit 4; }
done
if [ "$USED" -ge 1024 ] && [ "${SHARE_OWN:-0}" != 1 ]; then echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; fi
echo "=== disk ${USE}%  device $GPU $(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader) ${USED} MiB ${UTIL}%"
cd "$REPO" || exit 7

# 4. Each python that will run exists and sees the card.
need() { for m in "$@"; do P=$(py_for "$m"); [ -x "$P" ] || { echo "ABORT: no python for $m at $P" >&2; exit 2; }
  "$P" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: $P sees no CUDA" >&2; exit 4; }; done; }

run_method() {  # $1 method list joined by +, rest passed through
  local ms="$1"; shift; local m0="${ms%%+*}"
  "$(py_for "$m0")" 1-lora-correction/scripts/review_sd15_cell.py --method ${ms//+/ } "$@"
}

RC=0
case "$STAGE" in
  prep)
    need joint
    "$CO3" 1-lora-correction/scripts/review_sd15_prep.py || RC=1 ;;
  smoke)
    need joint aae syngen
    for ms in joint+poe aae dnb syngen; do
      echo "--- smoke $ms  $(date +%H:%M:%S)"
      run_method "$ms" --smoke || { echo "SMOKE $ms FAILED"; RC=1; }
    done
    echo "--- SynGen's own parse lines:"; grep -c "Final modifier indices collected" "/proc/$$/fd/1" 2>/dev/null || true
    "$CO3" 1-lora-correction/scripts/review_sd15_cell.py --check || RC=1 ;;
  check)
    "$CO3" 1-lora-correction/scripts/review_sd15_cell.py --check || RC=1 ;;
  run)
    : "${METHOD:?set METHOD}" "${SHARD:=0/1}"
    need ${METHOD//+/ }
    case "$METHOD" in joint+poe|joint|poe|aae|dnb|syngen) ;; *) echo "ABORT: METHOD $METHOD" >&2; exit 2;; esac
    run_method "$METHOD" --shard "$SHARD" || RC=1
    for m in ${METHOD//+/ }; do
      echo "=== $m: $(ls "$OUT/${m}_default/r1"/*/seed*/"$m".json 2>/dev/null | wc -l) sidecars of 1104 on disk"
    done ;;
  *) echo "ABORT: unknown STAGE $STAGE" >&2; exit 2 ;;
esac
echo "=== done $STAGE ${METHOD:-} ${SHARD:-}  rc $RC  $(date -Is)"
exit $RC
