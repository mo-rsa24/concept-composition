#!/usr/bin/env bash
# The tail of CO3's R1 and R3 renders for the ICLR 56173 review response (1-lora-correction/plans/the-baselines-the-reviewers-asked-for.md, task 3):
# the cells still missing when the six eval jobs slowed, given out as whole pairs, worked from the end of the list so
# the helpers meet the running jobs last. Shared-device path: one card, nohup over ssh.
#
#   ssh <node> 'GPU=<i> CELLS=<cells file> nohup bash <launchers>/review_baselines_co3_tail.sh > <log> 2>&1 < /dev/null & disown'
set -uo pipefail
HERE=/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers
export POE_REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
: "${GPU:?set GPU}" "${CELLS:?set CELLS}" "${CODE_COMMIT:=unknown}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== co3 tail  code $CODE_COMMIT  node $(hostname -s)  device $GPU  pid $$  cells $(wc -l < "$CELLS")  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$HERE/review_baseline_cell_v2.py" --method co3 --setting default --cells "$CELLS" || echo "!!! co3 tail FAILED"
echo "=== done $(date -Is)"
