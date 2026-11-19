#!/usr/bin/env bash
# The MNIST stage's 15 models, one after another on one card: train one (6,000 steps, the user's choice
# after the 6,000-step pilot), then draw its eight rows, then the next. Dial 0 first, so the dial-0 stop
# check can be read after the first model. Skips anything already saved or drawn, so it resumes.
# Usage, from the repo root: nohup bash poe_repair/experiments/inclusive_captions/mnist/run_all.sh > data/mnist/run.log 2>&1 &
set -u
PY=/home/molef/micromamba/envs/sdxl-local/bin/python
M=poe_repair/experiments/inclusive_captions/mnist
for d in 0 0.1 0.3 0.6 1; do
  for s in 1 2 3; do
    echo "=== dial $d seed $s · $(date +%H:%M)"
    $PY -I $M/train.py --only $d $s --steps 6000 2>&1 | grep -v -i -E 'autotune|addmm|precompil|^ +[0-9.]+ ms'
    $PY -I $M/sample.py --ckpt data/mnist/checkpoints/dial-$d-seed-$s.pt 2>&1 | grep -v -i -E 'autotune|addmm|precompil|^ +[0-9.]+ ms'
  done
done
echo "=== all done · $(date +%H:%M)"
