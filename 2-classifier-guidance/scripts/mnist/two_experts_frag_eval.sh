#!/bin/bash
#SBATCH --job-name=frag_eval
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster107
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/33-frag-eval-%j.log
# Plan 33, E2: the from-scratch fragments experts: each alone (strict and tolerant, bar 58 of 64 by the tolerant count), then composed on the five
# tuning pairs: mean-of-no-digit base, crop composition, and crop with both no-cross pushes; strict and tolerant counts.
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
M=$ROOT/two-experts/models; OUT=$ROOT/33-experts-at-inference; CK=$OUT/checkers; J=$ROOT/judge-v3/judge.pt
cd "$REPO"; S=2-classifier-guidance/scripts/mnist/two_experts.py
export CUDA_VISIBLE_DEVICES=0
TAG=${TAG:-frag}
$PY $S alone --expert $M/e04-$TAG/unet.pt --digits 0-4 --name e04-$TAG --out $OUT --judge $J
$PY $S alone --expert $M/e59-$TAG/unet.pt --digits 5-9 --name e59-$TAG --out $OUT --judge $J
EX="--a $M/e04-$TAG/unet.pt --b $M/e59-$TAG/unet.pt --judge $J --out $OUT"
H="--pair-head $CK/pair-nc2/head.pt --sep-head $CK/sep-nc/head.pt"
$PY $S cells $EX --base avgnull --push none --name $TAG-avgnull
$PY $S cells $EX --compose crop --push none --name $TAG-crop
$PY $S cells $EX --compose crop $H --push both --name $TAG-crop-both
echo "[$(date -u +%H:%M:%S)] done"
