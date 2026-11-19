#!/bin/bash
# Plan 33, E0d: whole canvas, layout inferred with the two regions forced disjoint; a (as predicted) and b (other half occluded); alone and with both no-cross pushes.
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/33-experts-at-inference; CK=$OUT/checkers; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
H="--pair-head $CK/pair-nc2/head.pt --sep-head $CK/sep-nc/head.pt"
EX="--a $M/${TAG:-e04-either}/unet.pt --b $M/${TAGB:-e59-either}/unet.pt --judge $J --out $OUT"
for c in inferred inferred-occ; do
  $PY $S cells $EX --compose $c --push none --name e0d-$c
  $PY $S cells $EX $H --compose $c --push both --name e0d-$c-both
done
echo "[$(date -u +%H:%M:%S)] done"
