#!/bin/bash
# Step M10 on a RunPod RTX 4090: one colour expert's ladder, the same arguments as the cluster's m7_train.sbatch in STEP=m10 mode
# (edm2-img64-xs, count labels on, batch 256 with 128 per pass, the ramp scaled to each run, 16 samples per count). The full run saves a
# checkpoint every quarter (500,224 images) so quarter_watch.py can sample and score it on the CPU while training continues.
# Usage: COLOUR=r|b [STAGES="overfit one-percent full"] [WAIT_PID=<pid>] bash run_ladder2.sh
set -uo pipefail
source /workspace/co3/bin/activate
W=/workspace/m10; E=$W/edm2; C=${COLOUR:?r or b}; STAGES=${STAGES:-"one-percent full"}
sample() {  # <out> <net>
  for K in 0 1 2; do
    (cd "$E" && python generate_images.py --net="$2" --outdir="$1/samples/count-$((K + 1))" --seeds=0-15 --class=$K --steps=32 --batch=16) \
      >> "$1/sample.log" 2>&1 || return 1
  done
}
run() {  # <rung> <set> <images seen> <checkpoint every>
  local rung=$1 set=$2 nimg=$3 every=$4 out=$W/runs/$1/$2 status net
  mkdir -p "$out"; status=$((nimg / 8 / 256 * 256)); [ "$status" -lt 256 ] && status=256
  if ! ls "$out"/network-snapshot-$(printf %07d $((nimg / 1000)))-*.pkl >/dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%S)] start $rung $set nimg=$nimg every=$every"
    (cd "$E" && M7_NIMG="$nimg" python "$W/scripts/edm2_scaled_ramp.py" --outdir="$out" --data="$W/data/$set.zip" --cond=True \
       --preset=edm2-img64-xs --duration="$nimg" --batch=256 --batch-gpu=128 --status="$status" --snapshot="$every" --checkpoint="$every" \
       --seed=20261124) >> "$out/train.log" 2>&1 || { echo "TRAIN FAILED $rung $set"; return 1; }
  fi
  net=$(ls -t "$out"/network-snapshot-*.pkl | head -1)
  [ -d "$out/samples" ] || sample "$out" "$net" || { echo "SAMPLE FAILED $rung $set"; return 1; }
  echo "[$(date -u +%H:%M:%S)] done $rung $set net=$net"
}
if [ -n "${WAIT_PID:-}" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done; fi
for s in $STAGES; do
  case $s in
    overfit)     run overfit "$C-overfit-18" 256000 256000 || exit 1 ;;
    one-percent) run one-percent "$C-one-percent" 65536 65536 || exit 1 ;;
    full)        run full "$C-full" 2000896 500224 || exit 1 ;;
  esac
done
echo "[$(date -u +%H:%M:%S)] LADDER DONE $C"
