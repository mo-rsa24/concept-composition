#!/usr/bin/env bash
# PoE + Langevin on a named cell list for the best- and worst-cases grids (scripts/review_langevin_cells.py).
#   sbatch --partition=bigbatch ... or ssh <node> 'GPU=<i> CELLS=<file> CODE_COMMIT=<sha> nohup bash <launchers>/review_langevin_cells.sh > <log> 2>&1 < /dev/null & disown'
set -uo pipefail
HERE=/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers
export POE_REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY="${PY:-/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python}"   # co3_bw on a Blackwell card
: "${GPU:=0}" "${CELLS:?set CELLS}" "${CODE_COMMIT:=unknown}"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== langevin cells  code $CODE_COMMIT  node $(hostname -s)  device $GPU  pid $$  cells $(wc -l < "$CELLS")  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$HERE/review_langevin_cells_v2.py" --cells "$CELLS" || echo "!!! langevin cells FAILED"
echo "=== done $(date -Is)"
