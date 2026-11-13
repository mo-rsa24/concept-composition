#!/bin/bash
#SBATCH --job-name=andsearch
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster65,mscluster74,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/and_vector_search/logs/%x-%A_%a.out
#
# Search for a third expert that puts two separate animals back in the frame, one shard per task.
# The excluded nodes are the bigbatch machines whose GPU is dead; a job landing there dies on the
# guard below in one second.
#
#   sbatch --array=0-5 5-text-embedding-third-expert/scripts/and_vector_search.sbatch
#   COLLECT=1 sbatch 5-text-embedding-third-expert/scripts/and_vector_search.sbatch        once the array is done
#
set -uo pipefail
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
OUT=${OUT:-/datasets/mmolefe/poe_repair_min/outputs/and_vector_search}
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
S=/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet/and_vector/_scripts
SHARD=${SLURM_ARRAY_TASK_ID:-0}
OF=${OF:-6}
SEEDS=${SEEDS:-"1 2 3 4"}

mkdir -p "$OUT/logs"
echo "=== node $(hostname -s) job ${SLURM_JOB_ID:-} shard $SHARD of $OF $(date -Is)"

# The guard the faulted nodes exist for: nvidia-smi answers but the device is unusable.
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || { echo "ABORT: no GPU" >&2; exit 4; }
pct=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "${pct:-100}" -lt 90 ] || { echo "ABORT: $OUT filesystem at ${pct}%" >&2; exit 3; }

export POE_REPO=$REPO PYTHONPATH=$REPO/scripts
cd "$REPO" || exit 7
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 5)" \
  || { echo "ABORT: torch sees no CUDA device" >&2; exit 5; }

if [ -n "${COLLECT:-}" ]; then
  "$PY" "$S/and_vector_search.py" --collect --out "$OUT"
else
  "$PY" "$S/and_vector_search.py" --shard "$SHARD" --of "$OF" --seeds $SEEDS --out "$OUT"
fi
echo "=== done $(date -Is) exit $?"
