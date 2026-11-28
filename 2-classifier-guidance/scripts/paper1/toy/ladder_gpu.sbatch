#!/bin/bash
#SBATCH --job-name=loss-ladder
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/toys/paper1/logs/%x-%j.log
#
# The loss ladder and the harder toy on one bigbatch GPU: runs loss_ladder.py or harder_toy.py (the first argument) with the
# remaining arguments from the cluster checkout. Both import squares_toy.py; this script imports the version committed at the
# checkout's HEAD (copied out with git show), so uncommitted edits to squares_toy.py by another session never reach a run.
# toy_gpu.sbatch's disk and GPU guards, unchanged. Submit with the exclude list read fresh each time:
#
#   sbatch --exclude=$(cat 2-classifier-guidance/scripts/paper1/bigbatch-exclude.txt) \
#       2-classifier-guidance/scripts/paper1/toy/ladder_gpu.sbatch loss_ladder.py ladder --world toy
#
# Slurm does not create the log folder; it must exist before submitting.
set -uo pipefail
[ $# -ge 1 ] || { echo "ABORT: no script given (loss_ladder.py or harder_toy.py)"; exit 2; }
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
TOY=2-classifier-guidance/scripts/paper1/toy
ROOT=/datasets/mmolefe/poe_repair_min/toys/paper1
PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python

# --- checks before any work ---
[ -d "$ROOT" ] || { echo "ABORT: $ROOT not visible from $(hostname -s)"; exit 3; }
mkdir -p "$ROOT/logs" /datasets/mmolefe/poe_repair_min/toys/paper1-harder
# The disk guard reads the filesystem this job writes to.
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python $PY missing on $(hostname -s)"; exit 3; }
# The GPU guard: nvidia-smi must list a device, and torch must reach it.
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)
[ -n "$GPU" ] || { echo "ABORT: nvidia-smi lists no GPU on $(hostname -s)"; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 4)" \
  || { echo "ABORT: torch cannot use the GPU ($GPU) on $(hostname -s)"; exit 4; }
cd "$REPO" || { echo "ABORT: no checkout at $REPO"; exit 3; }
COMMIT=$(git rev-parse --short HEAD)
# The committed squares_toy.py, four folders below /datasets so its REPO = parents[4] line resolves.
SRC=$ROOT/loss-ladder/src/$COMMIT
mkdir -p "$SRC"
git show HEAD:$TOY/squares_toy.py > "$SRC/squares_toy.py" || { echo "ABORT: cannot copy the committed squares_toy.py"; exit 3; }
export SQUARES_TOY_SRC=$SRC SQUARES_TOY_OUT=$ROOT SQUARES_TOY_RESULTS=$ROOT
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$GPU disk=${USE}% python=$PY commit=$COMMIT args=$*"

SCRIPT=$1
shift
"$PY" "$TOY/$SCRIPT" "$@"
rc=$?
echo "[$(date -u +%H:%M:%S)] EXIT $rc"
exit $rc
