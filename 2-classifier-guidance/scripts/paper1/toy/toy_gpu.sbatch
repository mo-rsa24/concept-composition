#!/bin/bash
#SBATCH --job-name=squares-toy
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/toys/paper1/logs/%x-%j.log
#
# Paper 1's squares toy on one bigbatch GPU: runs squares_toy.py with this script's arguments (a subcommand and its options) from
# the cluster checkout, with checkpoints, samples, results.json and pictures under /datasets/mmolefe/poe_repair_min/toys/paper1.
# Submit from the laptop through scripts/cluster.sh, with the exclude list on the sbatch line (it lists the bigbatch nodes whose
# GPUs have faulted):
#
#   scripts/cluster.sh run 'sbatch --job-name=toy-rung6 --exclude=$(cat 2-classifier-guidance/scripts/paper1/bigbatch-exclude.txt) \
#       2-classifier-guidance/scripts/paper1/toy/toy_gpu.sbatch rung6'
#
# Slurm does not create the log folder; it must exist before submitting (mkdir -p /datasets/mmolefe/poe_repair_min/toys/paper1/logs).
set -uo pipefail
[ $# -ge 1 ] || { echo "ABORT: no squares_toy.py subcommand given"; exit 2; }
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/datasets/mmolefe/poe_repair_min/toys/paper1
PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python

# --- checks before any work ---
[ -d "$ROOT" ] || { echo "ABORT: $ROOT not visible from $(hostname -s)"; exit 3; }
mkdir -p "$ROOT/logs"
# The disk guard reads the filesystem this job writes to.
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python $PY missing on $(hostname -s)"; exit 3; }
# The GPU guard: nvidia-smi must list a device, and torch must reach it (a job that cannot see the card runs on the CPU with no
# error, environment/hpc/nodes.md).
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)
[ -n "$GPU" ] || { echo "ABORT: nvidia-smi lists no GPU on $(hostname -s)"; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 4)" \
  || { echo "ABORT: torch cannot use the GPU ($GPU) on $(hostname -s)"; exit 4; }
cd "$REPO" || { echo "ABORT: no checkout at $REPO"; exit 3; }
export SQUARES_TOY_OUT=$ROOT SQUARES_TOY_RESULTS=$ROOT
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$GPU disk=${USE}% python=$PY commit=$(git rev-parse --short HEAD) args=$*"

"$PY" 2-classifier-guidance/scripts/paper1/toy/squares_toy.py "$@"
rc=$?
echo "[$(date -u +%H:%M:%S)] EXIT $rc"
exit $rc
