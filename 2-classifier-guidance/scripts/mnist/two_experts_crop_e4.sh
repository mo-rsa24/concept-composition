#!/bin/bash
# Plan 33, E4 proper on the crop composition of plan 32's experts. Checkers retrained with no cross-range pair in their pasted data
# (PC_NOCROSS=1): separate-label, pair-aware first pass, then two on-policy passes on pushed renders of the five tuning pairs
# (seeds 1000 up, 48 per pair), then crop + both pushes and crop + ink on the five tuning pairs, seeds 0 to 63.
# Usage: CUDA_DEV=0 setsid nohup bash two_experts_crop_e4.sh > <log> 2>&1 &
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/33-experts-at-inference; CK=$OUT/checkers; J=$ROOT/judge-v3/judge.pt
mkdir -p $OUT/draws $CK
cd "$REPO"; D=2-classifier-guidance/scripts/mnist
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0} PC_NOCROSS=1
echo "[$(date -u +%H:%M:%S)] start, commit $(git rev-parse --short HEAD)"
tr() { $PY $D/pixel_checker.py train --mnist $ROOT/mnist-raw --out $CK/$1 --kind $2 "${@:3}"; }
tr sep-nc separate
tr pair-nc0 pair
EX="--a $M/e04-either/unet.pt --b $M/e59-either/unet.pt --compose crop --judge $J --n 48"
draws() { $PY $D/two_experts.py draws $EX --push both --pair-head $CK/$1/head.pt --sep-head $CK/sep-nc/head.pt --draws-out $OUT/draws/$2.npz; }
draws pair-nc0 d1; tr pair-nc1 pair --draws $OUT/draws/d1.npz
draws pair-nc1 d2; tr pair-nc2 pair --draws $OUT/draws/d1.npz $OUT/draws/d2.npz
H="--pair-head $CK/pair-nc2/head.pt --sep-head $CK/sep-nc/head.pt --ink-head $ROOT/two-experts/checkers/ink/head.pt"
cell() { $PY $D/two_experts.py cells $EX $H --push $1 --name crop-nc-$1 --out $OUT; }
cell none; cell zero; cell both; cell ink
echo "[$(date -u +%H:%M:%S)] done"
