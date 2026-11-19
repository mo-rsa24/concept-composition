#!/bin/bash
# Plan 33, tuning round 1 on the crop composition with the no-cross checkers: one knob per run.
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/33-experts-at-inference; CK=$OUT/checkers; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
H="--pair-head $CK/pair-nc2/head.pt --sep-head $CK/sep-nc/head.pt"
EX="--a $M/e04-either/unet.pt --b $M/e59-either/unet.pt --compose crop --push both --judge $J --out $OUT $H"
t() { $PY $S cells $EX --name crop-nc-both-$1 "${@:2}"; }
t s3 --s 3; t s30 --s 30; t cap0.3 --cap 0.3; t eta0.5 --eta 0.5
echo "[$(date -u +%H:%M:%S)] done"
