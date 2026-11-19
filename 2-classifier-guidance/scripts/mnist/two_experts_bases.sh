#!/bin/bash
# Plan 32: the fixed-side plain product under more bases (the blank point mass and two small spreads). Usage: CUDA_DEV=0 nohup bash two_experts_bases.sh > <log> 2>&1 &
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/two-experts; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
c() { $PY $S cells --a $M/e04-left/unet.pt --b $M/e59-right/unet.pt --push none --base $1 --tau ${2:-0} --name fixed-none-$1${2:-} --out $OUT --judge $J; }
c blank; c floor 0.1; c floor 0.03
echo "[$(date -u +%H:%M:%S)] done"
