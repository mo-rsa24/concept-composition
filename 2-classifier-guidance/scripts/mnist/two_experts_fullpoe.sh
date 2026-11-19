#!/bin/bash
# Plan 33: the goal experiment. The full-canvas product of two experts (mean of the two no-digit predictions as base, no crop, no masks, no side
# assignment) with both no-cross pushes, and with both pushes plus the Langevin corrector, for three expert pairs. Five tuning pairs, seeds 0-63.
# Usage: TAGS="either frag ft-frag" CUDA_DEV=0 setsid nohup bash two_experts_fullpoe.sh > <log> 2>&1 &
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/33-experts-at-inference; CK=$OUT/checkers; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
H="--pair-head $CK/pair-nc2/head.pt --sep-head $CK/sep-nc/head.pt"
for tag in ${TAGS:-either frag ft-frag}; do
  EX="--a $M/e04-$tag/unet.pt --b $M/e59-$tag/unet.pt --judge $J --out $OUT --base avgnull --compose product $H --push both"
  $PY $S cells $EX --name fullpoe-$tag-both
  $PY $S cells $EX --lang-k 5 --lang-c 3 --name fullpoe-$tag-both-lang
done
echo "[$(date -u +%H:%M:%S)] done"
