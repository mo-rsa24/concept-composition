#!/usr/bin/env bash
# Evaluation renders for the ICLR 56173 review response (1-lora-correction/plans/the-baselines-the-reviewers-asked-for.md): one baseline at its
# published default on R1 (the 276 benchmark prompts, seeds 9 to 12, 1,104 cells) and R3 (the 53 held-out pairs,
# seeds 9 to 16, 424 cells), 1,528 cells in all, split into NSHARDS interleaved shards so R1 and R3 fill together.
#
#   METHOD=co3 SHARD=0 NSHARDS=2 sbatch --export=ALL --partition=bigbatch --nodes=1 --time=2-00:00:00 \
#     --exclude=<faulted list> --output=<OUT>/logs/eval_%x_%j.log <launchers>/review_baselines_eval.sh
#
# Writes <OUT>/<method>_default/{r1,r3}/<pair slug>/seedNN/<method>.png and .json. A tile that exists is skipped,
# so a relaunch with the same NSHARDS resumes its shard.
set -uo pipefail
# sbatch runs a copy of this file from the Slurm spool, so the files it reads are named by absolute path.
HERE=/datasets/mmolefe/poe_repair_min/outputs/paper_review/launchers
export POE_REPO=/home-mscluster/mmolefe/Playground/PhD/poe_repair_min
OUT=/datasets/mmolefe/poe_repair_min/outputs/paper_review
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
: "${METHOD:?set METHOD to co3 or superdiff}" "${SHARD:?set SHARD}" "${NSHARDS:?set NSHARDS}"

echo "=== eval $METHOD default shard $SHARD of $NSHARDS  node $(hostname -s)  device $GPU  pid $$  job ${SLURM_JOB_ID:-none}  $(date -Is)"
USE=$(df --output=pcent "$OUT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $OUT is ${USE:-unknown}% full" >&2; exit 3; }
[ -x "$PY" ] || { echo "ABORT: python not found at $PY" >&2; exit 2; }
read -r USED UTIL < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',') \
  || { echo "ABORT: no GPU $GPU" >&2; exit 4; }
case "$UTIL" in ''|*[!0-9]*) echo "ABORT: device $GPU faulted ($UTIL)" >&2; exit 4;; esac
[ "$USED" -lt 1024 ] || { echo "ABORT: device $GPU already holds ${USED} MiB" >&2; exit 4; }
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "ABORT: torch sees no CUDA" >&2; exit 4; }
[ -f "$OUT/latents/seed16.pt" ] || { echo "ABORT: no saved latents under $OUT/latents" >&2; exit 5; }
nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader

ALL=$(mktemp)
grep -v '^#' "$HERE/review_benchmark_pairs.txt" | awk -F'|' 'NF>=3 {for (s=9; s<=12; s++) print "r1|"$1"|"$2"|"$3"|"s}' > "$ALL"
grep -v '^#' "$HERE/review_held_out_pairs.txt" | awk -F'|' 'NF>=2 {for (s=9; s<=16; s++) print "r3|"$1"|"$2"|and|"s}' >> "$ALL"
N1=$(grep -c '^r1|' "$ALL"); N3=$(grep -c '^r3|' "$ALL")
[ "$N1" -eq 1104 ] && [ "$N3" -eq 424 ] || { echo "ABORT: $N1 R1 cells and $N3 R3 cells, not 1104 and 424" >&2; exit 2; }
CELLS="$OUT/logs/cells_${METHOD}_${SHARD}of${NSHARDS}.txt"
awk -v n="$NSHARDS" -v k="$SHARD" '(NR-1)%n==k' "$ALL" > "$CELLS"
rm -f "$ALL"
echo "=== $(wc -l < "$CELLS") cells in $CELLS"

"$PY" "$HERE/review_baseline_cell.py" --method "$METHOD" --setting default --cells "$CELLS" || echo "!!! $METHOD shard $SHARD FAILED"
echo "    tiles r1 $(ls "$OUT/${METHOD}_default"/r1/*/seed*/"$METHOD".png 2>/dev/null | wc -l) of 1104, r3 $(ls "$OUT/${METHOD}_default"/r3/*/seed*/"$METHOD".png 2>/dev/null | wc -l) of 424"
echo "=== done $(date -Is)"
