#!/bin/bash
# Plan 32: identity check (zero push equals no push, byte for byte), then one tuning round of both pushes on the two tuning pairs.
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/two-experts; CK=$OUT/checkers; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; D=2-classifier-guidance/scripts/mnist
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
$PY - <<PYEOF
import numpy as np
a = np.load("$OUT/cells/side-none-guided-e59-either/canvases.npz"); b = np.load("$OUT/cells/side-zero-guided-e59-either/canvases.npz")
print("identity: zero push equals no push on every pair:", all((a[k] == b[k]).all() for k in a.files))
PYEOF
H="--pair-head $CK/pair-r2/head.pt --sep-head $CK/separate/head.pt --ink-head $CK/ink/head.pt"
EX="--a $M/e04-either/unet.pt --b $M/e59-either/unet.pt --base floor --tau 0.5 --judge $J --pairs 2-8,0-5 --push both"
t() { $PY $D/two_experts.py cells $EX $H --name tune-$1 --out $OUT ${@:2}; }
t default; t s3 --s 3; t s30 --s 30; t cap0.3 --cap 0.3; t eta0.5 --eta 0.5
echo "[$(date -u +%H:%M:%S)] done"
