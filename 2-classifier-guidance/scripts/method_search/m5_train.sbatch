#!/bin/bash
#SBATCH --job-name=l6-m5-train
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82,mscluster84
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M5: train the clean-pair checker on the current or the balanced list. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> m5_train.sbatch <smoke|overfit|tiny|full> <current|balanced>
# Every stage first rebuilds the two lists and checks them against the committed composition.json.
set -uo pipefail
STAGE=${1:?stage}; LIST=${2:?list}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
OUT=$ROOT/m5
CO3=/datasets/mmolefe/poe_repair_min/loop-5/code/67b0bafa-b91076f/upstream/co3
: "${SNAP:?SNAP must name the code snapshot}"
REPO=$SNAP/repo
HUB=$SNAP/hub/scripts/reproductions
S=$REPO/2-classifier-guidance/scripts
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac
case "$LIST" in current|balanced) ;; *) echo "ABORT: unknown list $LIST"; exit 2 ;; esac
[ -f "$S/rung2/train_readout_pair.py" ] && [ -f "$S/method_search/build_balanced_list.py" ] && [ -x "$PY" ] || { echo "ABORT: snapshot or python missing"; exit 3; }
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
DEV=${GPU:-0}   # the device to use; 0 under Slurm, the approved free card on a shared-device run
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$DEV" | tr -d ',')
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$DEV" | wc -l)
# Under Slurm the card must hold no process. On an approved shared device (SHARED=1) another user's idle context may stay, so only the
# 1 GB memory limit applies (execution-protocol.md step 3), and the launcher's watchdog covers the rest of the run.
[ "$USED" -lt 1024 ] && { [ "$APPS" -eq 0 ] || [ "${SHARED:-0}" = 1 ]; } || { echo "ABORT: device $DEV in use: ${USED} MiB, ${APPS} processes"; exit 4; }
export CUDA_VISIBLE_DEVICES=$DEV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$HUB:$CO3:$REPO/scripts/shared:$REPO"
# The readout checker's latents, their augmented copies and its evaluation tiles live only under the scratch root, where the kept
# checker was trained; the trainer writes to --out-root under loop-6, so nothing is written there.
export POE_REPAIR_MOUNT=/home-mscluster/mmolefe/poe_repair_min_scratch
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) device=$DEV gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$DEV") pid=$$ snap=$SNAP stage=$STAGE list=$LIST"
mkdir -p "$OUT/lists"
"$PY" "$S/method_search/build_balanced_list.py" --out "$OUT/lists" --summary "$OUT/lists/composition.json" > /dev/null || { echo "ABORT: lists did not build"; exit 5; }
cmp -s "$OUT/lists/composition.json" "$REPO/artifacts/notes/progress/method-search/01b-balance-the-checker-data/composition.json" \
  || { echo "ABORT: the rebuilt lists differ from the committed composition.json"; exit 5; }
COMMON=(--manifest "$OUT/lists/training_set_$LIST.json" --pair-prompt --aug --early-frac 0.5 --accum 4 --out-root "$OUT/$STAGE")
case "$STAGE" in
  smoke) "$PY" "$S/rung2/train_readout_pair.py" "${COMMON[@]}" --run "$LIST" --steps 2 --eval-every 1000; rc=$? ;;
  overfit)
    # 16 of the list's pictures, checked on themselves; passes at a balanced accuracy of 0.9375 (15 of 16) at some step of the first 20.
    "$PY" "$S/method_search/m5_overfit_tiles.py" --list "$OUT/lists/training_set_$LIST.json" --out "$OUT/overfit/$LIST-tiles" || exit 5
    "$PY" "$S/rung2/train_readout_pair.py" --manifest "$OUT/overfit/$LIST-tiles/overfit-train.json" --eval "$OUT/overfit/$LIST-tiles/overfit-check.json" \
      --pair-prompt --aug --early-frac 0.5 --accum 4 --out-root "$OUT/overfit" --run "$LIST" --steps 400 --eval-every 100; rc=$?
    [ "$rc" -eq 0 ] && { "$PY" -c "
import json,sys; c=json.load(open('$OUT/overfit/$LIST/check.json')); b=[r['whole_evaluation_set_balanced_accuracy'] for r in c['per_step']]
best=max(b); print('OVERFIT', 'PASSED' if best>=0.9375 else 'FAILED', 'best balanced accuracy', round(best,4), 'at step', b.index(best)); sys.exit(0 if best>=0.9375 else 6)"; rc=$?; } ;;
  tiny)
    # 1% of the list's pictures per source and label; passes when the mean of the last two logged losses is below the first.
    "$PY" "$S/rung2/train_readout_pair.py" "${COMMON[@]}" --run "$LIST" --frac 0.01 --steps 400 --eval-every 400 | tee "$OUT/$STAGE-$LIST.log"; rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && { "$PY" -c "
import re,sys; l=[float(m) for m in re.findall(r'loss ([0-9.]+)', open('$OUT/$STAGE-$LIST.log').read())]
ok=len(l)>=3 and sum(l[-2:])/2<l[0]; print('LOSS', l, 'FALLS' if ok else 'DOES NOT FALL'); sys.exit(0 if ok else 6)"; rc=$?; } ;;
  full)  "$PY" "$S/rung2/train_readout_pair.py" "${COMMON[@]}" --run "$LIST" --steps 2000 --eval-every 500; rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE list=$LIST exit=$rc"
exit $rc
