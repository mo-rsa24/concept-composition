#!/bin/bash
#SBATCH --job-name=fixed-x
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/fixed_x/%x-%j.out
# mkdir -p /datasets/mmolefe/poe_repair_min/outputs/fixed_x before sbatch.
set -euo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/fixed_x
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
export POE_REPO="$REPO" PYTHONPATH="$REPO:$REPO/scripts" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="$OUT/tmp" XDG_CACHE_HOME="$OUT/cache"
export HF_HOME="${HF_HOME:-/datasets/mmolefe/.cache/huggingface}"
export TORCH_HOME="$OUT/cache/torch"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
"$PY" scripts/and_vector_fixed_x.py --out "$OUT" "$@"
