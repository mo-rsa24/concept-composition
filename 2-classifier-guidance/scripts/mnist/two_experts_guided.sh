#!/bin/bash
# Plan 32: the side-mixed guided cells. The pair-aware pixel checker gets two on-policy passes (its own pushed renders on the
# training pairs, seeds 1000 up, judged by the third judge, added to its training), then the default-setting cells
# (s 10, cap 0.1, eta 1.0) on the five pairs, seeds 0 to 63: zero push (identity check), random, both pushes, one-half ink.
# Base: the blank canvas with spread 0.5 (the fixed-side control's best base).
# Usage: E59=e59-either [or e59-either-s1] CUDA_DEV=0 nohup bash two_experts_guided.sh > <log> 2>&1 &
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/two-experts; CK=$OUT/checkers; J=$ROOT/judge-v3/judge.pt
A=${E04:-e04-either}; B=${E59:-e59-either}
cd "$REPO"; D=2-classifier-guidance/scripts/mnist
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
echo "[$(date -u +%H:%M:%S)] experts $A and $B, commit $(git rev-parse --short HEAD)"
EX="--a $M/$A/unet.pt --b $M/$B/unet.pt --base floor --tau 0.5 --judge $J"
draws() { $PY $D/two_experts.py draws $EX --push both --pair-head $CK/$1/head.pt --sep-head $CK/separate/head.pt --draws-out $OUT/draws/$2.npz; }
train() { $PY $D/pixel_checker.py train --kind pair --mnist $ROOT/mnist-raw --out $CK/$1 --draws "${@:2}"; }
mkdir -p $OUT/draws
draws pair-r0 draws1;  train pair-r1 $OUT/draws/draws1.npz
draws pair-r1 draws2;  train pair-r2 $OUT/draws/draws1.npz $OUT/draws/draws2.npz
H="--pair-head $CK/pair-r2/head.pt --sep-head $CK/separate/head.pt --ink-head $CK/ink/head.pt"
cell() { $PY $D/two_experts.py cells $EX $H --push $1 --name side-$1-guided-$B --out $OUT; }
cell none; cell zero; cell random; cell both; cell ink
echo "[$(date -u +%H:%M:%S)] done"
