#!/bin/bash
#SBATCH --job-name=prze2-identity
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster59,mscluster62,mscluster65,mscluster68,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster83
#SBATCH --output=/datasets/mmolefe/poe_repair_min/outputs/paper_review/logs/%x-%j.out
#
# PRZE-2 piece 3: the two new correctors at k = 0 against the plain samplers. First the stand-in
# UNet test (CPU, seconds), then one real SDXL cell, cat x dog at seed 10, compared with
# torch.equal. Writes nothing but this log. The plan is plans/prze-2-langevin-corrector.md.
#
#   scripts/cluster.sh run "sbatch scripts/check_corrector_identity.sbatch"
#   sbatch --export=ALL,REPO=<a worktree> <that worktree>/scripts/check_corrector_identity.sbatch
set -uo pipefail
REPO="${REPO:-/home-mscluster/mmolefe/Playground/PhD/poe_repair_min}"
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-none}  start $(date -Is)"
[ -d /datasets/mmolefe/poe_repair_min/outputs ] || { echo "ABORT: /datasets not visible here" >&2; exit 3; }
read -r USED UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ',')
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: gpu faulted ($UTIL)" >&2; exit 4;; esac
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 6)" || { echo "ABORT: torch sees no CUDA" >&2; exit 6; }
echo "=== gpu ${USED} MiB ${UTIL}%  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd "$REPO" || exit 7
echo "--- stand-in UNet test"
"$PY" -m pytest -q tests/test_corrector_drivers_identity.py; T=$?
echo "--- real SDXL, one cell"
"$PY" scripts/check_corrector_identity.py --pair a_cat__x__a_dog --seed 10; S=$?
echo "=== done $(date -Is)  pytest exit $T  sdxl exit $S"
exit $(( T || S ))
