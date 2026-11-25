#!/bin/bash
#SBATCH --job-name=l6-m7-data
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82,mscluster84,mscluster48,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M7: CPU work on the renders. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> m7_data.sbatch <control|diagnose|footprint|scorer|experts|paste|paste-score|pack> [set] [count] [zip name]
set -uo pipefail
STAGE=${1:?stage}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
: "${SNAP:?SNAP must name the code snapshot}"
S=$SNAP/repo/2-classifier-guidance/scripts/method_search/clevr_data.py
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
[ -f "$S" ] && [ -x "$PY" ] || { echo "ABORT: snapshot or python missing on $(hostname -s)"; exit 3; }
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) snap=$SNAP stage=$STAGE"
case "$STAGE" in
  control)  "$PY" "$S" control --renders "$ROOT/m7/full/renders" --out "$ROOT/m7/full/controls"; rc=$? ;;
  diagnose) "$PY" "$S" diagnose --renders "$ROOT/m7/full/renders" --out "$ROOT/m7/full/controls"; rc=$? ;;
  footprint) "$PY" "$S" footprint --windows tall-p3 --renders "$ROOT/m7/full/renders" --out "$ROOT/m7/full/controls"; rc=$? ;;
  paste)    "$PY" "$(dirname "$S")/clevr_paste.py" paste --samples "$ROOT/m7/full/samples" --out "$ROOT/m7/full/paste" --windows tall-p3; rc=$? ;;
  paste-score) "$PY" "$(dirname "$S")/clevr_paste.py" score --pasted "$ROOT/m7/full/paste/tall-p3" --scorer "$ROOT/m7/full/scorer"; rc=$? ;;
  experts)  "$PY" "$(dirname "$S")/clevr_scorer.py" experts --samples "$ROOT/m7/full/samples" --out "$ROOT/m7/full/scorer"; rc=$? ;;
  pack)     "$PY" "$S" pack --renders "$ROOT/m7/full/renders" --out "$ROOT/m7/full/data" --set "${2:?set}" --count "${3:-5000}" --name "${4:-$2}"; rc=$? ;;
  pack-s2)  # step M10: each colour expert's labelled dataset, plus the 16-picture and 1% subsets for its ladder
    R2=$ROOT/m7/s2/renders; D2=$ROOT/m10/data
    for C in r b; do
      "$PY" "$S" pack-labelled --renders "$R2" --out "$D2" --sets "${C}1,${C}2,${C}3" --count 5000 --name "$C-full" || exit 5
      "$PY" "$S" pack-labelled --renders "$R2" --out "$D2" --sets "${C}1,${C}2,${C}3" --count 50 --name "$C-one-percent" || exit 5
    done
    "$PY" "$S" pack-labelled --renders "$R2" --out "$D2" --sets "r1,r2,r3" --count 6 --name "r-overfit-18"; rc=$? ;;
  m10-scorer) # step M10: the red and blue count scorer, overfit on 16 real renders, then trained on 3,200 and held out on 800 against 95%
    O=$ROOT/m10/${SCORER_DIR:-scorer}; mkdir -p "$O"
    "$PY" "$(dirname "$S")/m10_scorer.py" overfit --renders "$ROOT/m7/s2/renders" --out "$O"; rc=$?
    [ "$rc" -eq 0 ] && { "$PY" "$(dirname "$S")/m10_scorer.py" train --renders "$ROOT/m7/s2/renders" --out "$O"; rc=$?; } ;;
  scorer)
    export WANDB__SERVICE_WAIT=300
    "$PY" "$(dirname "$S")/clevr_scorer.py" overfit --renders "$ROOT/m7/full/renders" --out "$ROOT/m7/full/scorer"; rc=$?
    [ "$rc" -eq 0 ] && { "$PY" "$(dirname "$S")/clevr_scorer.py" train --renders "$ROOT/m7/full/renders" --out "$ROOT/m7/full/scorer"; rc=$?; } ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
