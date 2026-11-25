#!/bin/bash
# Loop 12 step Z11 on a RunPod RTX 4090, from the payload built for it (z11-pod-payload.tar, unpacked at $P).
#   P=/workspace/z11 PY=<co3 python> bash z11_pod.sh <train-penalised|train-zero|train-shuffled|held|novel>
# The card must be an RTX 4090 so every Z11 row shares one card type; novel redraws the plain product's seen pictures here.
set -uo pipefail
STAGE=${1:?stage}; : "${P:?P must name the unpacked payload}"; PY=${PY:-python}
GPUNAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
case "$GPUNAME" in *"RTX 4090"*) ;; *) echo "ABORT: card is $GPUNAME; Z11 on pods runs on RTX 4090 only"; exit 4 ;; esac
S=$P/code/repo/2-classifier-guidance/scripts/leakage
export PYTHONPATH="$P/pylib:$P/code/repo:$P/code/repo/scripts/shared" TORCH_HOME=$P/torchhome HF_HUB_OFFLINE=1
ARGS="--mnist $P/mnist-raw --split $P/code/repo/artifacts/notes/progress/leakage/09-dcfm-colored-mnist/split.json --experts $P/z9/experts --judge $P/z9/judge/judge.pt --z9 $P/z9 --out $P/out"
mkdir -p $P/out $P/logs
echo "[$(date -u +%H:%M:%S)] host=$(hostname) gpu=$GPUNAME stage=$STAGE" | tee -a $P/logs/$STAGE.log
case "$STAGE" in
  train-*) $PY $S/z11_gradient_checker.py train --kind "${STAGE#train-}" $ARGS 2>&1 | tee -a $P/logs/$STAGE.log ;;
  held)    $PY $S/z11_gradient_checker.py held $ARGS 2>&1 | tee -a $P/logs/$STAGE.log ;;
  novel)   $PY $S/z11_gradient_checker.py novel --plain-here $ARGS 2>&1 | tee -a $P/logs/$STAGE.log ;;
  *) echo "ABORT: unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date -u +%H:%M:%S)] DONE stage=$STAGE exit=${PIPESTATUS[0]}" | tee -a $P/logs/$STAGE.log
