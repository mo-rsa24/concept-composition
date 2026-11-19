#!/bin/bash
# Plan 33, E2c: the post-hoc clutter fine-tuned experts of plan 32, each alone (bar 58 of 64), then composed on the five tuning pairs:
# mean-of-no-digit base, and the crop composition; strict and clutter-tolerant counts.
# Usage: CUDA_DEV=0 setsid nohup bash two_experts_ft_eval.sh > <log> 2>&1 &
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/33-experts-at-inference; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
TAG=${TAG:-ft-frag}
$PY $S alone --expert $M/e04-$TAG/unet.pt --digits 0-4 --name e04-$TAG --out $OUT --judge $J
$PY $S alone --expert $M/e59-$TAG/unet.pt --digits 5-9 --name e59-$TAG --out $OUT --judge $J
EX="--a $M/e04-$TAG/unet.pt --b $M/e59-$TAG/unet.pt --judge $J --out $OUT"
$PY $S cells $EX --base avgnull --push none --name $TAG-avgnull
$PY $S cells $EX --compose crop --push none --name $TAG-crop
echo "[$(date -u +%H:%M:%S)] done"
