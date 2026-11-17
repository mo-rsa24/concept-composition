#!/bin/bash
#SBATCH --job-name=r2_bench
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster68,mscluster74,mscluster75,mscluster76,mscluster83
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/outputs/classifier_guidance_ladder/rung2/logs/benchmark-slurm-%j.log
# One queue worker for plan 22's benchmark renders: claims pairs from the shared list until none
# are left. Writes to the scratch folder because /datasets is down (plan 22, Considerations).
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
[ -x "$PY" ] || { echo "co3 python missing at $PY"; exit 2; }
cd /home-mscluster/mmolefe/Playground/PhD/concept-composition || exit 2
export POE_REPAIR_MOUNT=/home-mscluster/mmolefe/poe_repair_min_scratch
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
exec "$PY" 2-classifier-guidance/scripts/rung2/render_benchmark.py --queue
