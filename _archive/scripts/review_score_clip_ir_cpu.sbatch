#!/bin/bash
# Score one batch of paper-review renders with CLIP score and ImageReward on CPU (protocol, "The scorers").
#   sbatch --export=ALL,BATCH=batch1 scripts/review_score_clip_ir_cpu.sbatch     (reads $S/rows_$BATCH.csv)
#SBATCH -p batch
#SBATCH -c 6
#SBATCH --mem=28G
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -J score-clip-ir
#SBATCH -o /datasets/mmolefe/poe_repair_min/outputs/paper_review/scores/logs/score_%x_%j.log
set -euo pipefail
S=/datasets/mmolefe/poe_repair_min/outputs/paper_review/scores
cd $S
USE=$(df --output=pcent $S | tail -1 | tr -dc '0-9'); [ "$USE" -lt 90 ] || { echo "ABORT: $S ${USE}% full"; exit 3; }
hostname; nproc
CO3=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
T2I=/datasets/mmolefe/envs/t2icomp/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6
echo "== CLIP score and ImageReward on CPU $(date -Is)"
$CO3 bin/review_score_clip_ir.py --in rows_$BATCH.csv --out clip_ir_$BATCH.csv
echo "== done $(date -Is)"
