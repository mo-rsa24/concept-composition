# paper2_guards.sh: the checks every Paper 2 script opens with, sourced, never run on its own.
#
# Paper 2's runs go inside one interactive allocation on one Blackwell node (mscluster110, 111 or
# 112, one 96 GB card), as plain `bash <script>` from the cluster checkout, sharing device 0 with
# Paper 1's process. No sbatch, no SSH launch onto another node. The checks follow step 4 of
# environment/hpc/execution-protocol.md, adjusted for a shared card:
#
#   p2_disk_guard DIR      df on the filesystem DIR sits on; stop at 90% full (environment/storage.md)
#   p2_python              PY: co3_bw on mscluster110 to 112, co3 elsewhere (environment/hpc/nodes.md)
#   p2_gpu_guard MIN_MIB   device 0 answers, is not faulted ([N/A], "requires reset"), and has at least
#                          MIN_MIB free; another process on it is allowed (Paper 1 shares the card)
#   p2_cuda_probe PY       torch.cuda.is_available() and one matmul on the card: under the wrong
#                          environment a Blackwell card runs on the CPU with no error
#   p2_hf_env              HF_HOME on /datasets, offline
#   p2_co3_path            PYTHONPATH for CO3 as its rows load it: CO3's upstream code, the subject
#                          guard, the repo; checks co3_common.py is in method_search/
#   p2_log RUN             everything after this line also goes to $P2_ROOT/logs/RUN.log
#
# Each failed check prints ABORT and the reason, and exits with its own code: 3 disk, 4 GPU,
# 5 python, 6 CUDA, 7 a missing file or folder.

P2_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
P2_SCRIPTS="$P2_REPO/2-classifier-guidance/scripts/paper2"
P2_ROOT=/datasets/mmolefe/poe_repair_min/paper2
P2_HF_HOME=/datasets/mmolefe/.cache/huggingface
P2_VLM_ENV="${P2_VLM_ENV:-/datasets/mmolefe/envs/paper2-vlm}"
# CO3's own code (upstream/co3, its composers package) comes from the code snapshot the Paper 2 rows already
# read (2-classifier-guidance/scripts/method_search/m1_paper2.sbatch). Its loader, co3_common.py, is imported
# from method_search/ in this repo (paper2_rows.py puts that folder first); it was restored there on 2026-11-26
# and is byte-identical to the hub repository's tracked copy (replications/published-methods-on-our-pairs,
# commit b91076f, the commit in the snapshot's name).
P2_CO3_SNAP="${P2_CO3_SNAP:-/datasets/mmolefe/poe_repair_min/loop-5/code/67b0bafa-b91076f}"

p2_abort() { echo "ABORT: $2"; exit "$1"; }

p2_disk_guard() {
  local dir="$1" use
  mkdir -p "$dir" || p2_abort 3 "cannot create $dir on $(hostname -s)"
  use=$(df --output=pcent "$dir" | tail -1 | tr -dc '0-9')
  [ -n "$use" ] && [ "$use" -lt 90 ] || p2_abort 3 "$dir is ${use:-unknown}% full"
  echo "disk: $dir on $(df --output=target "$dir" | tail -1) is ${use}% full"
}

p2_python() {
  case "$(hostname -s)" in
    mscluster11[0-2]) PY=/home-mscluster/mmolefe/miniforge3/envs/co3_bw/bin/python ;;
    *)                PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python ;;
  esac
  [ -x "$PY" ] || p2_abort 5 "no python at $PY"
  echo "python: $PY"
}

p2_gpu_guard() {
  local min="$1" used total util pstate
  command -v nvidia-smi >/dev/null || p2_abort 4 "no nvidia-smi on $(hostname -s)"
  IFS=',' read -r used total util pstate < <(nvidia-smi -i 0 \
    --query-gpu=memory.used,memory.total,utilization.gpu,pstate --format=csv,noheader,nounits 2>&1 | tr -d ' ')
  case "$used$total" in ''|*[!0-9]*) p2_abort 4 "device 0 memory reads '$used' of '$total': missing or faulted";; esac
  case "$util" in ''|*[!0-9]*) p2_abort 4 "device 0 utilisation reads '$util': faulted";; esac
  case "$pstate" in *eset*|*N/A*) p2_abort 4 "device 0 power state reads '$pstate'";; esac
  [ $((total - used)) -ge "$min" ] || p2_abort 4 "device 0 has $((total - used)) MiB free, under the $min MiB this run needs"
  echo "gpu: $(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader), ${used} of ${total} MiB used, ${util}% busy," \
       "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l) other processes; needs $min MiB"
}

p2_cuda_probe() {
  "$1" -c "import torch,sys; ok=torch.cuda.is_available() and (torch.ones(1000,1000,device='cuda')@torch.ones(1000,1000,device='cuda')).sum().item()==1e9; print('cuda probe:', torch.__version__, torch.cuda.get_device_name(0) if ok else 'no card'); sys.exit(0 if ok else 6)" \
    || p2_abort 6 "CUDA probe failed under $1: wrong environment for this card, or the card is faulted"
}

p2_hf_env() {
  export HF_HOME="$P2_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  [ -d "$HF_HOME" ] || p2_abort 7 "HF_HOME $HF_HOME is not visible on $(hostname -s)"
}

p2_co3_path() {
  local loader="$P2_REPO/2-classifier-guidance/scripts/method_search/co3_common.py" co3="$P2_CO3_SNAP/upstream/co3"
  [ -f "$loader" ] || p2_abort 7 "co3_common.py not found at $loader (CO3's row, the dial trainer and the load check import it)"
  [ -d "$co3/composers" ] || p2_abort 7 "CO3's code not found at $co3"
  [ -f "$P2_REPO/scripts/shared/disallowed_subjects.py" ] || p2_abort 7 "subject guard missing from $P2_REPO"
  export PYTHONPATH="$co3:$P2_REPO/scripts/shared:$P2_REPO${PYTHONPATH:+:$PYTHONPATH}"
  echo "co3: loader $loader, code $co3"
}

p2_log() {
  mkdir -p "$P2_ROOT/logs"
  local log="$P2_ROOT/logs/$1.log"
  exec > >(tee -a "$log") 2>&1
  echo "[$(date -u +%FT%TZ)] run=$1 node=$(hostname -s) repo=$P2_REPO commit=$(git -C "$P2_REPO" rev-parse --short HEAD 2>/dev/null) log=$log"
}
