#!/bin/bash
# Plan 33, E3 round 1: Langevin corrector on the crop composition with both no-cross pushes (the review file gives the rule).
# Usage: LK=5 LC=3 CUDA_DEV=0 setsid nohup bash two_experts_langevin.sh > <log> 2>&1 &
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/33-experts-at-inference; CK=$OUT/checkers; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
H="--pair-head $CK/pair-nc2/head.pt --sep-head $CK/sep-nc/head.pt"
EX="--a $M/e04-either/unet.pt --b $M/e59-either/unet.pt --compose crop --push both --judge $J --out $OUT $H"
$PY $S cells $EX --lang-k 0 --lang-c 0 --name crop-nc-both-k0check --pairs 4-7
$PY $S cells $EX --lang-k ${LK:-5} --lang-c ${LC:-3} --name crop-nc-both-lang-k${LK:-5}-c${LC:-3}
echo "[$(date -u +%H:%M:%S)] done"
