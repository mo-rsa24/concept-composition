#!/bin/bash
# Plan 33, the final read: the chosen setting once on the other 20 cross-range pairs, seeds 64 to 127, plus the crop composition without pushes as a reference.
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/33-experts-at-inference; CK=$OUT/checkers; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
P20=0-6,0-7,0-8,0-9,1-5,1-6,1-7,1-8,2-5,2-6,2-7,2-9,3-5,3-7,3-8,3-9,4-5,4-6,4-8,4-9
H="--pair-head $CK/pair-nc2/head.pt --sep-head $CK/sep-nc/head.pt"
EX="--a $M/e04-either/unet.pt --b $M/e59-either/unet.pt --compose crop --judge $J --out $OUT --pairs $P20 --eval-seed0 64"
$PY $S cells $EX --push none --name final-crop-none
$PY $S cells $EX $H --push both --lang-k 5 --lang-c 3 --name final-crop-both-lang5c3
echo "[$(date -u +%H:%M:%S)] done"
