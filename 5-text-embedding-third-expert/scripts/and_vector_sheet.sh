#!/bin/bash
# The "and" vector sheet on one shared biggpu device, launched over SSH outside Slurm:
#
#   ssh mscluster106 'GPU=1 nohup bash /datasets/.../and_vector/_scripts/and_vector_sheet.sh \
#       > /datasets/.../and_vector/logs/sheet.log 2>&1 &'
#
# Renders the tiles, then the same-prompts norm check. Guards per environment/hpc/execution-protocol.md.
set -uo pipefail
GPU=${GPU:?set GPU to the device index}
REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/concept_pair_sheet
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8"}
SET=${SET:-vectors}

echo "=== node $(hostname -s)  device $GPU  pid $$  set $SET  seeds $SEEDS  $(date -Is)"
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits) || { echo "ABORT: no device $GPU" >&2; exit 4; }
[ "$used" -lt 1024 ] || { echo "ABORT: device $GPU has ${used} MiB in use" >&2; exit 4; }
[ -x "$PY" ] || { echo "ABORT: $PY missing" >&2; exit 5; }
pct=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ "${pct:-100}" -lt 90 ] || { echo "ABORT: $OUT filesystem at ${pct}%" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES=$GPU POE_REPO=$REPO CONCEPT_PAIR_OUT=$OUT PYTHONPATH=$REPO/scripts
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 5)" || { echo "ABORT: torch sees no CUDA device" >&2; exit 5; }

cd "$REPO" || exit 7
"$PY" "$HERE/and_vector_sheet.py" --render --set "$SET" --a dog --b cat --joint "dog and cat" --seeds $SEEDS || exit 8
if [ "$SET" = vectors ]; then
  "$PY" "$HERE/and_vector_norms.py" --joint "dog and cat" --a dog --b cat --out "$OUT/../and_vector_norms" || exit 9
fi
echo "=== done $(date -Is)"
