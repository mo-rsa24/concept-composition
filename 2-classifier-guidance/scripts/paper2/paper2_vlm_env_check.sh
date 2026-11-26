#!/bin/bash
# paper2_vlm_env_check.sh: the vision-language environment loads Qwen3-VL-8B on the card and answers one question.
#
#   bash 2-classifier-guidance/scripts/paper2/paper2_vlm_env_check.sh [--crop <png>]
#
# After paper2_vlm_env_build.sh and task 6. Run from the cluster checkout on the held Blackwell node, with
# the environment's own python, HF_HOME on /datasets and HF_HUB_OFFLINE=1. Opens with the disk guard, the
# card guard (about 20 GB free: the 8B model in bf16 is about 17 GB) and the CUDA probe. Writes
# outputs/paper2/paper2-vlm-env/load.txt in the checkout and a copy beside the log.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paper2_guards.sh"
cd "$P2_REPO" || exit 7
p2_log paper2-vlm-env
p2_disk_guard "$P2_ROOT/paper2-vlm-env"
PY="$P2_VLM_ENV/bin/python"
[ -x "$PY" ] || p2_abort 5 "no environment at $P2_VLM_ENV: run paper2_vlm_env_build.sh first"
echo "python: $PY"
p2_hf_env
ls -d "$HF_HOME/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"*/ >/dev/null 2>&1 \
  || p2_abort 7 "Qwen3-VL-8B-Instruct is not in $HF_HOME: run task 6 (paper2_fetch_weights.sh) first"
p2_gpu_guard 20000
export CUDA_VISIBLE_DEVICES=0
p2_cuda_probe "$PY"
"$PY" "$P2_SCRIPTS/paper2_vlm_load_check.py" --out outputs/paper2/paper2-vlm-env/load.txt "$@"
rc=$?
[ -f outputs/paper2/paper2-vlm-env/load.txt ] && cp outputs/paper2/paper2-vlm-env/load.txt "$P2_ROOT/paper2-vlm-env/load.txt"
[ $rc -eq 0 ] && echo "PASS: Qwen3-VL-8B answered on the card" || echo "FAIL: exit $rc"
exit $rc
