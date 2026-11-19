#!/bin/bash
# Plan 33, E2c: plan 32's either-side experts, trained 4,000 more steps on their own digit plus fragment clutter, one after the other.
# Usage: CUDA_DEV=0 setsid nohup bash two_experts_finetune.sh > <log> 2>&1 &
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models
cd "$REPO"
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
for spec in "0-4 e04-either e04-ft-frag" "5-9 e59-either e59-ft-frag"; do
  set -- $spec
  $PY 2-classifier-guidance/scripts/mnist/canvas_unet.py train --mnist $ROOT/mnist-raw --share 0 --out $M --digits $1 --side either \
      --tag $3 --amp16 --clutter-mode fragments --init $M/$2/unet.pt --steps 4000 --save-every 4000
done
echo "[$(date -u +%H:%M:%S)] done"
