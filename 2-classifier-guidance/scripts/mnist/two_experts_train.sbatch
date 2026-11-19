#!/bin/bash
#SBATCH --job-name=two_experts
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster107
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas/logs/two-experts-%j.log
# Plan 32: the four digit experts on one biggpu node, two processes per card (mscluster107: 2x Quadro RTX 8000).
# Usage: sbatch scripts/mnist/two_experts_train.sbatch time              200 steps: fp32, fp16, two fp16 sharing card 0
#        sbatch scripts/mnist/two_experts_train.sbatch train [flags]     the four experts, e.g. train --amp16
#        SPECS="0-4 left e04-left 0;5-9 right e59-right 0" bash <this file> train --amp16
#            a chosen set of experts ("digits side tag card", ';' between), also outside Slurm (mscluster85)
# Timing on mscluster107 (job 90405): 210 ms/step fp32, 148 fp16, 316 each for two fp16 sharing a card.
set -euo pipefail
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/home-mscluster/mmolefe/poe_repair_min_scratch/mnist-canvas
OUT=$ROOT/two-experts/models
MODE=${1:?usage: two_experts_train.sbatch time|train [flags]}; shift
cd "$REPO"
[ -x "$PY" ] || { echo "co3 python missing at $PY"; exit 2; }
[ -f "$ROOT/mnist-raw/train-images-idx3-ubyte" ] || { echo "MNIST raw files missing under $ROOT/mnist-raw"; exit 2; }
mkdir -p "$OUT" "$ROOT/logs"
USED=$(df --output=pcent "$OUT" | tail -1 | tr -dc 0-9)
[ -n "$USED" ] && [ "$USED" -lt 90 ] || { echo "disk guard: $OUT at ${USED:-unknown}%"; exit 3; }
nvidia-smi -L | grep -q GPU || { echo "GPU guard: no device visible"; exit 4; }
echo "[$(date -u +%H:%M:%S)] node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,) disk=${USED}% commit=$(git rev-parse --short HEAD) mode=$MODE args=$*"
T="$PY 2-classifier-guidance/scripts/mnist/canvas_unet.py train --mnist $ROOT/mnist-raw --share 0 --out $OUT"

if [ "$MODE" = time ]; then
    CUDA_VISIBLE_DEVICES=0 $T --digits 0-4 --side either --tag timing --time 200
    CUDA_VISIBLE_DEVICES=0 $T --digits 0-4 --side either --tag timing --time 200 --amp16
    echo "[$(date -u +%H:%M:%S)] two fp16 processes sharing card 0"
    CUDA_VISIBLE_DEVICES=0 $T --digits 0-4 --side either --tag timing-shared-a --time 200 --amp16 &
    CUDA_VISIBLE_DEVICES=0 $T --digits 5-9 --side either --tag timing-shared-b --time 200 --amp16 &
    wait
else
    FLAGS="$*"
    pids=()
    IFS=';' read -ra LIST <<< "${SPECS:-0-4 either e04-either 0;5-9 either e59-either 0;0-4 left e04-left 1;5-9 right e59-right 1}"
    for spec in "${LIST[@]}"; do
        set -- $spec
        CUDA_VISIBLE_DEVICES=$4 $T --digits "$1" --side "$2" --tag "$3" $FLAGS > "$ROOT/logs/two-experts-$3-${SLURM_JOB_ID:-$(hostname -s)}.log" 2>&1 &
        pids+=($!)
        echo "  started $3 on card $4 (pid $!)"
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [ $fail = 0 ] || { echo "an expert failed; see the per-expert logs"; exit 5; }
fi
echo "[$(date -u +%H:%M:%S)] done"
