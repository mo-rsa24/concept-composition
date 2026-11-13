#!/usr/bin/env bash
# CO3 and SuperDiff at their published defaults on Figure 5's four pairs at the shared seed 10 (layout row R2), for
# the ICLR 56173 review response (1-lora-correction/plans/the-baselines-the-reviewers-asked-for.md, task 3). 8 tiles.
#
#   ssh <node> 'GPU=0 nohup bash <launchers>/review_baselines_fig5.sh > <OUT>/logs/fig5_baselines.log 2>&1 < /dev/null & disown'
#
# Writes <OUT>/<method>_default/r2/<pair slug>/seed10/<method>.png and .json. A tile that exists is skipped.
set -uo pipefail
HERE=/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers
export POE_REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
PAIRS=("a cat|a dog" "an elephant|a penguin" "a tent|a snowy mountain" "a lighthouse|a stormy sea")

echo "=== fig5 baselines  node $(hostname -s)  device $GPU  pid $$  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
[ -f "$OUT/latents/seed10.pt" ] || { echo "ABORT: no saved latent for seed 10" >&2; exit 5; }
nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader

for M in co3 superdiff; do
  echo "--- $M  $(date +%H:%M:%S)"
  "$PY" "$HERE/review_baseline_cell.py" --method "$M" --setting default --group r2 --pairs "${PAIRS[@]}" --seeds 10 \
    || echo "!!! $M FAILED"
done
echo "    tiles $(ls "$OUT"/{co3,superdiff}_default/r2/*/seed10/*.png 2>/dev/null | wc -l) of 8"
echo "=== done $(date -Is)"
