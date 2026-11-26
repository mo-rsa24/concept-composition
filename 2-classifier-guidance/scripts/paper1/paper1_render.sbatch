#!/bin/bash
#SBATCH --job-name=paper1-cpu
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=14:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster65,mscluster68,mscluster71,mscluster74,mscluster75,mscluster76,mscluster79,mscluster82,mscluster87
#SBATCH --output=/datasets/mmolefe/poe_repair_min/paper1/logs/%x-%j.log
#
# HOW IT RUNS NOW: on the owner's held Blackwell node (salloc on mscluster110, 111 or 112, one 96 GB card, co3_bw), as a plain bash
# command from the cluster checkout, so the checks below still run before any work; the #SBATCH lines are comments under bash:
#
#   bash 2-classifier-guidance/scripts/paper1/paper1_render.sbatch <run> <command...> 2>&1 | tee /datasets/mmolefe/poe_repair_min/paper1/logs/<run>.log
#
# The sbatch form below is kept for a later stage that goes back to the queue.
#
# Paper 1: one CPU run on a 28-thread bigbatch node. Blender 2.78c renders on the CPU only (job 94448), so renders take a node
# and no card. Runs the committed code in the cluster checkout. Submit from the cluster repo root, with the whole exclude list on
# the sbatch line (a command-line --exclude replaces the one above):
#
#   scripts/cluster.sh run 'sbatch --job-name=<run> --exclude=$(cat 2-classifier-guidance/scripts/paper1/bigbatch-exclude.txt) \
#       2-classifier-guidance/scripts/paper1/paper1_render.sbatch <run> <command> [+ <command>]...'
#
# A command is one of:
#   render <set> <red objects> <blue objects> <seed base> <start> <count>
#       joint CLEVR scenes with clevr_render.py's joint mode into /datasets/mmolefe/poe_repair_min/paper1/<run>/renders/<set>/,
#       per-object masks at 64 pixels beside them; scene i is seeded with <seed base> + i. "cycle cycle" for the counts gives
#       scene i (i mod 3) + 1 red and as many blue, the training pool's sizes in turn (2, 4, 6, 2, ... objects).
#   python <script> <args...>
#       a CPU python script, given by its path from the repository root (paste, packing, sheets, the model check); it must
#       carry --run <run>.
# Commands separated by a lone + run in order and stop at the first failure. At the end the run folder's top-level small files
# are copied to outputs/paper1/<run>/ in the cluster checkout.
set -uo pipefail
RUN=${1:?run name}; shift
[ $# -ge 1 ] || { echo "ABORT: no command given"; exit 2; }
REPO=${REPO:-/home-mscluster/mmolefe/Playground/PhD/concept-composition}
ROOT=/datasets/mmolefe/poe_repair_min/paper1
OUT=$ROOT/$RUN
SMALL=$REPO/outputs/paper1/$RUN
BL=/datasets/mmolefe/poe_repair_min/loop-5/third_party/blender-2.78c/blender-2.78c-linux-glibc219-x86_64/blender
export CLEVR_GEN=/datasets/mmolefe/poe_repair_min/loop-5/third_party/clevr-dataset-gen/image_generation
RENDER=$REPO/2-classifier-guidance/scripts/method_search/clevr_render.py

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
# The GPU guard. This job uses the CPU, so a busy card is allowed; a node whose card is missing or faulted is refused, since on
# bigbatch that has marked broken nodes (environment/hpc/nodes.md).
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ',')
case "$USED" in ''|*[!0-9]*) echo "ABORT: no GPU visible or memory reading '$USED'"; exit 4;; esac
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
export CUDA_VISIBLE_DEVICES=""
cd "$REPO" || { echo "ABORT: no checkout at $REPO"; exit 3; }
THREADS=$(nproc)
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) threads=$THREADS disk=${USE}% python=$PY commit=$(git rev-parse --short HEAD) run=$RUN"

# --- the commands ---
rc=0
while [ $# -gt 0 ]; do
  CMD=()
  while [ $# -gt 0 ] && [ "$1" != "+" ]; do CMD+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  case "${CMD[0]}" in
    render)
      [ "${#CMD[@]}" -eq 7 ] || { echo "ABORT: render needs <set> <red> <blue> <seed base> <start> <count>"; rc=2; break; }
      SET=${CMD[1]}; NRED=${CMD[2]}; NBLUE=${CMD[3]}; BASE=${CMD[4]}; START=${CMD[5]}; COUNT=${CMD[6]}
      JOINT="$NRED,$NBLUE"; [ "$NRED" = cycle ] && JOINT=cycle
      [ -x "$BL" ] && [ -f "$CLEVR_GEN/render_images.py" ] && [ -f "$RENDER" ] || { echo "ABORT: renderer missing on $(hostname -s)"; rc=3; break; }
      [ "$BASE" -ge 8000000 ] || { echo "ABORT: seed base $BASE is below 8,000,000, where earlier jobs rendered"; rc=2; break; }
      DEST=$OUT/renders
      for _ in 1 2 3 4 5 6; do mkdir -p "$DEST/$SET" 2>/dev/null; [ -d "$DEST/$SET" ] && break; sleep 5; done
      echo "[$(date -u +%H:%M:%S)] RENDER set=$SET red=$NRED blue=$NBLUE base=$BASE start=$START count=$COUNT"
      "$BL" --background --python-exit-code 7 --threads "$THREADS" --python "$RENDER" -- --name "$SET" --joint "$JOINT" \
        --seed-base "$BASE" --start "$START" --count "$COUNT" --out "$DEST" --masks 64 2>&1 \
        | grep --line-buffered -E 'M7RENDER|P1RENDER|Traceback|Error:|error:'
      rc=${PIPESTATUS[0]}
      MISSING=0
      for ((i = START; i < START + COUNT; i++)); do
        [ -f "$DEST/$SET/scenes/$(printf '%s_%06d.json' "$SET" "$i")" ] || MISSING=$((MISSING + 1))
      done
      echo "[$(date -u +%H:%M:%S)] EXIT $rc render set=$SET asked=$COUNT missing=$MISSING"
      [ "$rc" -eq 0 ] && [ "$MISSING" -ne 0 ] && rc=9 ;;
    python)
      [ -f "$REPO/${CMD[1]:-none}" ] || { echo "ABORT: no script $REPO/${CMD[1]:-}"; rc=2; break; }
      [[ " ${CMD[*]} " == *" --run $RUN "* || " ${CMD[*]} " == *" --run $RUN/"* ]] || { echo "ABORT: command does not carry --run $RUN or --run $RUN/<part>: ${CMD[*]}"; rc=2; break; }
      echo "[$(date -u +%H:%M:%S)] RUN ${CMD[*]:1}"
      "$PY" "$REPO/${CMD[1]}" "${CMD[@]:2}"; rc=$?
      echo "[$(date -u +%H:%M:%S)] EXIT $rc ${CMD[1]}" ;;
    *) echo "ABORT: unknown command ${CMD[0]}"; rc=2 ;;
  esac
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
