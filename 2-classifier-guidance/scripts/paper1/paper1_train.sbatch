#!/bin/bash
#SBATCH --job-name=paper1-train
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster71,mscluster74,mscluster75,mscluster76,mscluster79,mscluster82,mscluster87
#SBATCH --output=/datasets/mmolefe/poe_repair_min/paper1/logs/%x-%j.log
#
# HOW IT RUNS NOW: on the owner's held Blackwell node (salloc on mscluster110, 111 or 112, one 96 GB card, co3_bw), as a plain bash
# command from the cluster checkout, so the checks below still run before any work; the #SBATCH lines are comments under bash:
#
#   bash 2-classifier-guidance/scripts/paper1/paper1_train.sbatch <run> <command...> 2>&1 | tee /datasets/mmolefe/poe_repair_min/paper1/logs/<run>.log
#
# The sbatch form below is kept for a later stage that goes back to the queue.
#
# Paper 1: train one head (paper1_disc.py train, after its block choice) on a bigbatch RTX 3090, logging the loss to W&B project
# prime_lab/poe-repair-animals-compose. Runs the committed code in the cluster checkout. Submit from the cluster repo root, with
# the whole exclude list on the sbatch line (a command-line --exclude replaces the one above):
#
#   scripts/cluster.sh run 'sbatch --job-name=<run> --exclude=$(cat 2-classifier-guidance/scripts/paper1/bigbatch-exclude.txt) \
#       2-classifier-guidance/scripts/paper1/paper1_train.sbatch <run> paper1_disc.py <subcommand> --run <run> <args...> [+ ...]'
#
# Commands separated by a lone + run in order and stop at the first failure; each must carry --run <run>. The head writes its
# checkpoint under /datasets/mmolefe/poe_repair_min/paper1/<run>/; the top-level small files are copied to outputs/paper1/<run>/.
set -uo pipefail
RUN=${1:?run name}; shift
[ $# -ge 1 ] || { echo "ABORT: no command given"; exit 2; }
REPO=${REPO:-/home-mscluster/mmolefe/Playground/PhD/concept-composition}
ROOT=/datasets/mmolefe/poe_repair_min/paper1
OUT=$ROOT/$RUN
SMALL=$REPO/outputs/paper1/$RUN
CODE=$REPO/2-classifier-guidance/scripts/paper1
FREE_MIN_MIB=${FREE_MIN_MIB:-16000}   # free memory on device 0 this job needs

# --- checks before any work (environment/hpc/execution-protocol.md, step 4) ---
[ -d "$ROOT" ] || { echo "ABORT: $ROOT not visible from $(hostname -s)"; exit 3; }
for _ in 1 2 3 4 5 6; do mkdir -p "$OUT" 2>/dev/null; [ -d "$OUT" ] && break; sleep 5; done
[ -d "$OUT" ] || { echo "ABORT: could not create $OUT"; exit 3; }
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full"; exit 3; }
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
[ -x "$PY" ] || { echo "ABORT: python $PY missing on $(hostname -s)"; exit 3; }
# The card may be shared with Paper 2's process on the same held node, so a busy card is allowed; what is refused is a missing or
# faulted card, or less free memory than this job needs (FREE_MIN_MIB, set at the top).
read -r USED FREE UTIL TOTAL < <(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ',')
case "$USED" in ''|*[!0-9]*) echo "ABORT: no GPU visible or memory reading '$USED'"; exit 4;; esac
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)
echo "gpu device 0: used ${USED} MiB, free ${FREE} MiB, ${APPS} processes, needs ${FREE_MIN_MIB} MiB free"
[ "$FREE" -ge "$FREE_MIN_MIB" ] || { echo "ABORT: device 0 has ${FREE} MiB free, this job needs ${FREE_MIN_MIB}"; exit 4; }
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 WANDB__SERVICE_WAIT=300
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 6)" \
  || { echo "ABORT: CUDA probe failed"; exit 6; }
cd "$REPO" || { echo "ABORT: no checkout at $REPO"; exit 3; }
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) memory=${TOTAL}MiB disk=${USE}% python=$PY commit=$(git rev-parse --short HEAD) run=$RUN"

# --- the commands ---
rc=0
while [ $# -gt 0 ]; do
  CMD=()
  while [ $# -gt 0 ] && [ "$1" != "+" ]; do CMD+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  [ -f "$CODE/${CMD[0]}" ] || { echo "ABORT: no script $CODE/${CMD[0]}"; rc=2; break; }
  [[ " ${CMD[*]} " == *" --run $RUN "* || " ${CMD[*]} " == *" --run $RUN/"* ]] || { echo "ABORT: command does not carry --run $RUN or --run $RUN/<part>: ${CMD[*]}"; rc=2; break; }
  echo "[$(date -u +%H:%M:%S)] RUN ${CMD[*]}"
  "$PY" "$CODE/${CMD[0]}" "${CMD[@]:1}"; rc=$?
  echo "[$(date -u +%H:%M:%S)] EXIT $rc ${CMD[0]}"
  [ "$rc" -eq 0 ] || break
done

# --- small outputs to the cluster checkout ---
mkdir -p "$SMALL"
find "$OUT" -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' -o -name '*.csv' -o -name '*.png' -o -name '*.md' \) \
  -size -20M -exec cp {} "$SMALL"/ \;
# A blind sheet's folder travels too, so reviewers can pull it; its key.json stays at the run's top level, outside sheet/.
[ -d "$OUT/sheet" ] && cp -r "$OUT/sheet" "$SMALL"/
echo "[$(date -u +%H:%M:%S)] DONE run=$RUN exit=$rc small=$(ls "$SMALL" | wc -l) files in $SMALL"
exit $rc
