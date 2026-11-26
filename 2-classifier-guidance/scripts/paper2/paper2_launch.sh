#!/bin/bash
# paper2_launch.sh: run one Paper 2 GPU step on the held Blackwell node, behind the guards.
#
#   bash 2-classifier-guidance/scripts/paper2/paper2_launch.sh <run-name> <co3|vlm> <min-free-MiB> <script> [arguments]
#
# Run from the cluster checkout on the node the owner holds (mscluster110, 111 or 112). It checks the
# disk under /datasets/mmolefe/poe_repair_min/paper2/<run-name>, picks the python (co3: co3_bw on 110 to
# 112, co3 elsewhere; vlm: the vision-language environment's own python), checks device 0 has at least
# <min-free-MiB> free and is not faulted (Paper 1's process may share it), runs the CUDA probe, sets
# HF_HOME on /datasets offline and CO3's PYTHONPATH, pins CUDA_VISIBLE_DEVICES=0, and logs everything to
# /datasets/mmolefe/poe_repair_min/paper2/logs/<run-name>.log. <script> is a file name under
# 2-classifier-guidance/scripts/paper2/; the arguments pass through unchanged.
set -uo pipefail
RUN=${1:?run name}; WHICH=${2:?co3 or vlm}; MIN=${3:?minimum free MiB on device 0}; SCRIPT=${4:?script}; shift 4
source "$(dirname "${BASH_SOURCE[0]}")/paper2_guards.sh"
cd "$P2_REPO" || exit 7
p2_log "$RUN"
p2_disk_guard "$P2_ROOT/$RUN"
case "$WHICH" in
  co3) p2_python ;;
  vlm) PY="$P2_VLM_ENV/bin/python"; [ -x "$PY" ] || p2_abort 5 "no vision-language environment at $P2_VLM_ENV (build it with paper2_vlm_env_build.sh)"; echo "python: $PY" ;;
  *)   p2_abort 2 "second argument must be co3 or vlm, not '$WHICH'" ;;
esac
[ -f "$P2_SCRIPTS/$SCRIPT" ] || p2_abort 7 "no script $P2_SCRIPTS/$SCRIPT"
p2_gpu_guard "$MIN"
export CUDA_VISIBLE_DEVICES=0
p2_cuda_probe "$PY"
p2_hf_env
p2_co3_path
echo "[$(date -u +%FT%TZ)] start $SCRIPT $*"
"$PY" "$P2_SCRIPTS/$SCRIPT" "$@"
rc=$?
echo "[$(date -u +%FT%TZ)] done $SCRIPT exit=$rc"
exit $rc
