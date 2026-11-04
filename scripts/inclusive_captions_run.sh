#!/usr/bin/env bash
# Train one arm of the inclusive-captions pilot and render its watching sheet as each
# checkpoint lands, so the four columns fill in Weights and Biases while the run is still going.
#
#   scripts/inclusive_captions_run.sh <A|B> <cuda device index> [steps]
#
# biggpu allows one Slurm job per user, so both arms are started with nohup outside Slurm on a
# pinned free device and are invisible to squeue. Harvest them with pgrep on the node they were
# started on.
set -u

RUN="${1:?arm: A or B}"
DEVICE="${2:?cuda device index, e.g. 1}"
STEPS="${3:-2000}"

PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/inclusive_captions/$RUN
WANDB_ID=inclusive-captions-run-$RUN

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES="$DEVICE"
export PYTHONPATH="$REPO"

# The Blackwell nodes answer nvidia-smi and then run on the CPU without an error, so assert the
# device before committing hours to it.
"$PY" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('no CUDA device visible: this would run on the CPU')
print('device:', torch.cuda.get_device_name(0), flush=True)
" || exit 1

"$PY" -m poe_repair.experiments.inclusive_captions.trainer \
  --run "$RUN" --steps "$STEPS" --wandb-id "$WANDB_ID" &
TRAINER=$!
echo "trainer pid $TRAINER" >&2

# Render each checkpoint once, into the trainer's own W&B run. A directory is only ready when its
# weights file has stopped growing, because save_pretrained writes it in place.
drawn=""
last_pass=0
while :; do
    kill -0 "$TRAINER" 2>/dev/null || last_pass=1
    for ckpt in "$OUT"/step_*; do
        [ -d "$ckpt" ] || continue
        case " $drawn " in *" $ckpt "*) continue ;; esac
        weights="$ckpt/adapter_model.safetensors"
        [ -f "$weights" ] || continue
        a=$(stat -c %s "$weights"); sleep 5; b=$(stat -c %s "$weights")
        [ "$a" = "$b" ] || continue
        step="${ckpt##*step_}"
        echo "sheet for step $step" >&2
        "$PY" -m poe_repair.experiments.inclusive_captions.sheets \
          --adapter "$ckpt" --label "run $RUN, step $((10#$step))" \
          --wandb-id "$WANDB_ID"
        drawn="$drawn $ckpt"
    done
    # One more pass after the trainer exits, so the final checkpoint still gets its sheet.
    [ "$last_pass" = 1 ] && break
    sleep 60
done

wait "$TRAINER"
status=$?
echo "trainer exited $status" >&2
exit $status
