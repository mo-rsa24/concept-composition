#!/bin/bash
# Comp. (validated animal instance count) on CPU for the paper-review renders (protocol, "The scorers").
#   sbatch --export=ALL,ROWS="rows_batch1.csv rows_batch2.csv" scripts/review_score_comp_cpu.sbatch
#SBATCH -p batch
#SBATCH -c 6
#SBATCH --mem=28G
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J score-comp
#SBATCH -o /datasets/mmolefe/poe_repair_min/outputs/paper_review/scores/logs/score_%x_%j.log
set -euo pipefail
S=/datasets/mmolefe/poe_repair_min/outputs/paper_review/scores
cd $S
hostname; nproc
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 PYTHONPATH=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python bin/review_score_comp.py --in $ROWS --out comp.csv
echo "== done $(date -Is)"
