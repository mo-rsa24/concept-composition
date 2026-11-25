#!/bin/bash
#SBATCH --job-name=l6-m5-read
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --exclude=mscluster84,mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M5: read the kept checker on CO3's 96 tuning trajectories, then score the read. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> m5_read.sbatch <smoke|read>
set -uo pipefail
STAGE=${1:?stage}; SHARD=${2:-0}; NUM=${3:-1}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
OUT=$ROOT/m5/co3-read
M1=$ROOT/m1/paper-2/full/renders/co3
HEAD=/home-mscluster/mmolefe/poe_repair_min_scratch/outputs/classifier_guidance_ladder/rung2/readout_pair/readout-multipair/head_step_002000.pt
CO3=/datasets/mmolefe/poe_repair_min/loop-5/code/67b0bafa-b91076f/upstream/co3
: "${SNAP:?SNAP must name the code snapshot}"
REPO=$SNAP/repo
NOTES=$REPO/artifacts/notes/progress/method-search/01-beat-the-baseline
CALLS=$NOTES/paper-2-blind-calls.csv
KEY=$NOTES/paper-2-blind-key.json
HUB=$SNAP/hub/scripts/reproductions
SPLITS=$REPO/2-classifier-guidance/scripts/checker_choice/splits.json
S=$REPO/2-classifier-guidance/scripts/method_search
case "$(hostname -s)" in
  mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
esac

[ -d "$ROOT" ] && [ -d "$CO3" ] && [ -d "$M1" ] && [ -f "$CALLS" ] && [ -f "$KEY" ] && [ -f "$S/m5_read_co3.py" ] && [ -f "$S/paper2_rows.py" ] && [ -f "$HUB/co3_common.py" ] || { echo "ABORT: snapshot or $ROOT not visible on $(hostname -s)"; exit 3; }
[ -f "$HEAD" ] && [ -f "$SPLITS" ] || { echo "ABORT: missing checker or splits"; exit 3; }
[ -f "$REPO/scripts/shared/disallowed_subjects.py" ] || { echo "ABORT: subject guard missing from the snapshot"; exit 8; }
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
[ -x "$PY" ] || { echo "ABORT: no python at $PY"; exit 5; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "${GPU:-0}" | tr -d ',')
case "$USED" in ''|*[!0-9]*) echo "ABORT: gpu memory reading '$USED'"; exit 4;; esac
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted, utilisation '$UTIL'"; exit 4;; esac
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${GPU:-0}" | wc -l)
[ "$USED" -lt 1024 ] && { [ "$APPS" -eq 0 ] || [ "${SHARED:-0}" = 1 ]; } || { echo "ABORT: device 0 in use: ${USED} MiB, ${APPS} processes"; exit 4; }
export CUDA_VISIBLE_DEVICES=${GPU:-0} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and (torch.ones(100,100,device='cuda')@torch.ones(100,100,device='cuda')).sum().item()==1e6 else 6)" \
  || { echo "ABORT: CUDA probe failed"; exit 6; }
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0) python=$PY snap=$SNAP stage=$STAGE shard=$SHARD/$NUM"
export PYTHONPATH="$HUB:$CO3:$REPO/scripts/shared:$REPO"
mkdir -p "$OUT"
case "$STAGE" in
  smoke) "$PY" "$S/m5_read_co3.py" read --head "$HEAD" --splits "$SPLITS" --m1-co3 "$M1" --out "$OUT-smoke" --limit 1; rc=$? ;;
  read)  "$PY" "$S/m5_read_co3.py" read --head "$HEAD" --splits "$SPLITS" --m1-co3 "$M1" --out "$OUT"; rc=$?
         [ "$rc" -eq 0 ] && { "$PY" "$S/m5_read_co3.py" score --out "$OUT" --calls "$CALLS" --key "$KEY"; rc=$?; } ;;
  reread) "$PY" "$S/m5_read_co3.py" read --head "$HEAD" --splits "$SPLITS" --m1-co3 "$M1" --out "$OUT-rtx8000" --redo-from "$OUT/reads.jsonl"; rc=$?
         [ "$rc" -eq 0 ] && { "$PY" "$S/m5_read_co3.py" score --out "$OUT" --also "$OUT-rtx8000" --calls "$CALLS" --key "$KEY"; rc=$?; } ;;
  save)   # keep the read steps' states for one of M1's shards, on the card type that drew it: shard 0 a Quadro RTX 8000, shard 1 an RTX 3090
         CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU:-0}")
         case "$SHARD:$CARD" in 0:*"RTX 8000"*|1:*"RTX 3090"*) ;; *) echo "ABORT: shard $SHARD must run on the card type M1 used, not $CARD"; exit 7 ;; esac
         "$PY" "$S/m5_read_co3.py" read --head "$HEAD" --splits "$SPLITS" --m1-co3 "$M1" --out "$OUT-states-shard$SHARD" \
           --shard "$SHARD" --num-shards 2 --save-latents; rc=$? ;;
  control) # the kept checker alone on the saved states: its scores must match those recorded in the read
         "$PY" "$S/m5_read_co3.py" heads --heads "kept=$HEAD" --latents-from "$OUT-states-shard0" "$OUT-states-shard1" \
           --out "$ROOT/m5/heads-read-control" --calls "$CALLS" --key "$KEY"; rc=$? ;;
  perplace) # step M12: the per-place prediction difference on the saved states, and the one-step clean estimates for the judge
         "$PY" "$S/m5_read_co3.py" perplace --latents-from "$OUT-states-shard0" "$OUT-states-shard1" \
           --out "$ROOT/m12/perplace" --calls "$CALLS" --key "$KEY"; rc=$? ;;
  fresh)  # M12's second reader: CO3 at fresh seeds 6009-6016 on the 12 tuning pairs, states saved; SHARD=1 is the smoke (one trajectory)
         SC=$ROOT/m12/size-corrected; LIM=""; [ "$SHARD" = 1 ] && { SC=$SC-smoke; LIM="--limit 1"; }
         "$PY" "$S/m5_read_co3.py" read --head "$HEAD" --splits "$SPLITS" --seeds 6009-6016 --save-latents --out "$SC/co3" $LIM; rc=$?
         [ "$rc" -eq 0 ] && { "$PY" "$S/m5_read_co3.py" perplace --latents-from "$SC/co3" --out "$SC/perplace" --count; rc=$?; } ;;
  expected) # each subject drawn alone by plain SDXL, and its share of the picture measured by the detector
         "$PY" "$S/m5_read_co3.py" expected --splits "$SPLITS" --out "$ROOT/m12/size-corrected/expected"; rc=$? ;;
  heads)  # score the kept checker (the control) and M5's two trained checkers on the saved states
         F=$ROOT/m5/full
         "$PY" "$S/m5_read_co3.py" heads --heads "kept=$HEAD" "current=$F/current/head_step_002000.pt" "balanced=$F/balanced/head_step_002000.pt" \
           --latents-from "$OUT-states-shard0" "$OUT-states-shard1" --out "$ROOT/m5/heads-read" --calls "$CALLS" --key "$KEY"; rc=$? ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=$rc"
exit $rc
