#!/bin/bash
#SBATCH --job-name=score_compose
#SBATCH --partition=biggpu
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/home-mscluster/mmolefe/Playground/PhD/concept-composition/logs/score_compose_%j.log
# Step 10: score every saved epoch of phase1_r8_100k with the instance-count scorer, including
# steps 70000 to 100000, which were rendered during training but never scored. No rendering and
# no training: it reads samples/per_epoch/*/ and rewrites compose_rate.json, logging the curves
# to the run's own W&B run (1d3qy31e).
#
# The compose_rate.json that stops at step 60000 is copied to compose_rate.to-step-60000.json
# first, so the rescored early steps can be checked against it.
set -euo pipefail

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
RUN=$REPO/1-lora-correction/artifacts/results/does-it-reach-unseen-pairs/the-transfer/does-the-fix-reach-unseen-pairs/pooled_lora/phase1_r8_100k
cd "$REPO"

echo "[$(date -u +%H:%M:%S)] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[$(date -u +%H:%M:%S)] epochs on disk: $(ls "$RUN/samples/per_epoch" | wc -l)"

[ -f "$RUN/compose_rate.to-step-60000.json" ] || cp "$RUN/compose_rate.json" "$RUN/compose_rate.to-step-60000.json"
[ -f "$RUN/compose_rate.with-cmp-strips.json" ] || cp "$RUN/compose_rate.json" "$RUN/compose_rate.with-cmp-strips.json"

$PY -m poe_repair.experiments.does_the_fix_reach_unseen_pairs.compose_rate \
    --run-dir "$RUN" --wandb --run-id 1d3qy31e

echo "[$(date -u +%H:%M:%S)] done"
