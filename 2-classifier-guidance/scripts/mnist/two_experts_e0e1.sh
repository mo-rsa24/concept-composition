#!/bin/bash
# Plan 33: E0 region composition (full-canvas and half-crop variants) and E1 union base, plan 32's either-side experts,
# five tuning pairs, seeds 0 to 63, no training. Also the identity check: a zero push equals no push for the region composer.
# Usage: CUDA_DEV=0 setsid nohup bash two_experts_e0e1.sh > <log> 2>&1 &
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/33-experts-at-inference; J=$ROOT/judge-v3/judge.pt
mkdir -p $OUT
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
EX="--a $M/e04-either/unet.pt --b $M/e59-either/unet.pt --judge $J --out $OUT"
$PY $S cells $EX --compose region --push none --name e0-region
$PY $S cells $EX --compose region --push zero --name e0-region-zero
$PY $S cells $EX --compose crop --push none --name e0-crop
$PY $S cells $EX --base union --union $ROOT/models/share-0.0/unet.pt --push none --name e1-union
$PY - <<PYEOF
import numpy as np
a = np.load("$OUT/cells/e0-region/canvases.npz"); b = np.load("$OUT/cells/e0-region-zero/canvases.npz")
print("identity: zero push equals no push for region composition:", all((a[k] == b[k]).all() for k in a.files))
PYEOF
echo "[$(date -u +%H:%M:%S)] done"
