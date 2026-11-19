#!/bin/bash
# Plan 32: the fixed-side plain product under more bases. BASES="blank floor:2 floor:3" picks them (default: blank floor:0.1 floor:0.03). Usage: CUDA_DEV=0 nohup bash two_experts_bases.sh > <log> 2>&1 &
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/two-experts; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
c() { [ "$2" = "$1" ] && set -- $1 ""; $PY $S cells --a $M/e04-left/unet.pt --b $M/e59-right/unet.pt --push none --base $1 --tau ${2:-0} --name fixed-none-$1${2:-} --out $OUT --judge $J; }
for b in ${BASES:-blank floor:0.1 floor:0.03}; do c ${b%%:*} ${b##*:}; done
echo "[$(date -u +%H:%M:%S)] done"
