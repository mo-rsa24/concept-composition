#!/bin/bash
#SBATCH --job-name=invert-x
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/invert_x/%x-%j.out
# Before submission: mkdir -p /datasets/mmolefe/poe_repair_min/outputs/invert_x
set -euo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/invert_x
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
mkdir -p "$OUT"
export TMPDIR="$OUT/tmp" HF_HOME=/datasets/mmolefe/.cache/huggingface
export XDG_CACHE_HOME="$OUT/cache" TORCH_HOME="$OUT/cache/torch"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
export POE_REPO="$REPO" PYTHONPATH="$REPO:$REPO/scripts" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
"$PY" -c 'import torch; assert torch.cuda.is_available(), "no usable CUDA device"'
"$PY" scripts/and_vector_invert_x.py --out "$OUT" "$@"
