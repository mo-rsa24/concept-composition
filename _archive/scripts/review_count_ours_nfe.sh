#!/usr/bin/env bash
# One cell of ours with a U-Net counter, for protocol rule 6 (scripts/review_count_ours_nfe.py).
#   ssh <node> 'GPU=<i> CODE_COMMIT=<sha> nohup bash <launchers>/review_count_ours_nfe.sh > <log> 2>&1 < /dev/null & disown'
set -uo pipefail
HERE=/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers
export POE_REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
CKPT=/datasets/mmolefe/poe_repair_min/outputs/showcase/cohort3/v58-03-self/checkpoints/lora_step_030000.pt
: "${GPU:?set GPU}" "${CODE_COMMIT:=unknown}"
export CUDA_VISIBLE_DEVICES="$GPU" FIGURE_CANDIDATES_OUT="$OUT/nfe_count"
echo "=== count ours nfe  code $CODE_COMMIT  node $(hostname -s)  device $GPU  pid $$  $(date -Is)"
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
[ -f "$CKPT" ] || { echo "ABORT: no checkpoint at $CKPT" >&2; exit 2; }
cd "$POE_REPO" || exit 2
"$PY" "$HERE/review_count_ours_nfe.py" --column ours --pairs "a seal|a walrus" --seeds 11 --checkpoint "$CKPT" \
  --window 50 --tag count || echo "!!! count FAILED"
echo "=== done $(date -Is)"
