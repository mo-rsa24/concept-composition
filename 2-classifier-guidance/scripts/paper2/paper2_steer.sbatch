#!/bin/bash
#SBATCH --job-name=p2-steer-judge
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/datasets/mmolefe/poe_repair_min/paper2/logs/%x-%j.log
#
# The steering judge's GPU steps (paper2_steer_judge.py features, paper2_steer_vlm.py) on one bigbatch RTX 3090.
# Submit from the cluster repo root with the exclude list read fresh:
#
#   sbatch --exclude=$(cat 2-classifier-guidance/scripts/paper1/bigbatch-exclude.txt) \
#       2-classifier-guidance/scripts/paper2/paper2_steer.sbatch <env: co3_bw or vlm> <script> <args...>
#
# The checks run before any work: the run folder is visible and under 90% full, the python exists, and CUDA answers
# with one matmul on the card (a CPU fallback would run with no error and take hours).
set -uo pipefail
ENV=${1:?co3_bw or vlm}; shift
[ $# -ge 1 ] || { echo "ABORT: no command given"; exit 2; }
REPO=/home-mscluster/mmolefe/Playground/PhD/concept-composition
ROOT=/datasets/mmolefe/poe_repair_min/paper2
OUT=$ROOT/paper2-judge-steer
[ -d "$ROOT" ] || { echo "ABORT: $ROOT not visible from $(hostname -s)"; exit 3; }
mkdir -p "$OUT" || { echo "ABORT: cannot create $OUT"; exit 3; }
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full"; exit 3; }
echo "disk: $OUT is ${USE}% full"
case "$ENV" in
  co3_bw) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
  vlm)    PY=/datasets/mmolefe/envs/paper2-vlm/bin/python ;;
  *)      echo "ABORT: unknown environment $ENV"; exit 5 ;;
esac
[ -x "$PY" ] || { echo "ABORT: no python at $PY"; exit 5; }
"$PY" -c "import torch; assert torch.cuda.is_available(); x = torch.ones(512, 512, device='cuda'); print('cuda:', torch.cuda.get_device_name(0), float((x @ x)[0, 0]))" \
  || { echo "ABORT: CUDA does not answer under $PY"; exit 6; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
export HF_HOME=/datasets/mmolefe/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO/2-classifier-guidance/scripts/paper2" || exit 7
echo "host $(hostname -s), commit $(git -C "$REPO" rev-parse --short HEAD), $(date)"
"$PY" "$@"
echo "exit $? at $(date)"
