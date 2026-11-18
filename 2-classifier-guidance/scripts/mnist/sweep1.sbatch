#!/bin/bash
#SBATCH --job-name=mnist_sweep1
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster74,mscluster75,mscluster76,mscluster82,mscluster83
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/sweep1-%j.log
# Plan 24 task 1.7, sweep 1: each weaker checker's pair-aware push on 7 & 8, seeds 0 to 63, s = 0, 1, 3, 10, 30.
# Usage: sbatch [--nodelist=<node>] [--dependency=...] scripts/mnist/sweep1.sbatch <checker name> [<checker name> ...]
set -euo pipefail
cd /home-mscluster/mmolefe/Playground/PhD/concept-composition
for h in "$@"; do
  bash 2-classifier-guidance/scripts/mnist/five_versions.sbatch "sweep1-$h" --pairs 78 --pair-only --pair-head "$h"
done
