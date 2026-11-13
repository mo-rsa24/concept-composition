#!/bin/bash
# Score one batch of paper-review renders with BLIP-VQA on an RTX 3090 (protocol, "The scorers").
#   sbatch --export=ALL,BATCH=batch1 1-lora-correction/scripts/review_score_blip.sbatch     (reads $S/rows_$BATCH.csv)
#SBATCH -p bigbatch
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -J score-blip
#SBATCH --exclude=mscluster68,mscluster88,mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster59,mscluster62,mscluster65,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster83
#SBATCH -o /datasets/mmolefe/poe_repair_min/outputs/paper_review/scores/logs/score_%x_%j.log
set -euo pipefail
S=/datasets/mmolefe/poe_repair_min/outputs/paper_review/scores
cd $S
USE=$(df --output=pcent $S | tail -1 | tr -dc '0-9'); [ "$USE" -lt 90 ] || { echo "ABORT: $S ${USE}% full"; exit 3; }
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
CO3=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
T2I=/datasets/mmolefe/envs/t2icomp/bin/python
$T2I -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "ABORT: no CUDA"; exit 4; }
echo "== BLIP-VQA $(date -Is)"
rm -rf work_blip_$BATCH
$T2I bin/review_score_blipvqa.py --in rows_$BATCH.csv --out blip_$BATCH.csv --work work_blip_$BATCH
echo "== done $(date -Is)"
