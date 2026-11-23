#!/bin/bash
#SBATCH --job-name=l6-m7-render
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --time=14:00:00
#SBATCH --exclude=mscluster44,mscluster45,mscluster50,mscluster51,mscluster57,mscluster61,mscluster65,mscluster68,mscluster71,mscluster72,mscluster73,mscluster74,mscluster75,mscluster76,mscluster82
#SBATCH --output=/datasets/mmolefe/poe_repair_min/loop-6/logs/%x-%j.log
# Step M7: render one CLEVR set on the CPU. Submit with the code snapshot in SNAP:
#   sbatch --export=ALL,SNAP=<snapshot> m7_render.sbatch <rung> <set> <start> <count>
# set: p1, p2, p3, empty, or s0 to s7, where the bits of sN choose which of the three positions hold an object.
set -uo pipefail
RUNG=${1:?rung}; SET=${2:?set}; START=${3:?start}; COUNT=${4:?count}
ROOT=/datasets/mmolefe/poe_repair_min/loop-6
OUT=$ROOT/m7/$RUNG/renders
BL=/datasets/mmolefe/poe_repair_min/loop-5/third_party/blender-2.78c/blender-2.78c-linux-glibc219-x86_64/blender
export CLEVR_GEN=/datasets/mmolefe/poe_repair_min/loop-5/third_party/clevr-dataset-gen/image_generation
: "${SNAP:?SNAP must name the code snapshot}"
S=$SNAP/repo/2-classifier-guidance/scripts/method_search/clevr_render.py
P1="0.30,0.30"; P2="0.60,0.30"; P3="0.40,0.60"
case "$SET" in
  p1) TARGETS="$P1"; BASE=1000000 ;;
  p2) TARGETS="$P2"; BASE=2000000 ;;
  p3) TARGETS="$P3"; BASE=3000000 ;;
  empty) TARGETS=""; BASE=4000000 ;;
  s[0-7])
    M=${SET#s}; TARGETS=""; BASE=$((5000000 + M * 100000))
    [ $((M & 1)) -ne 0 ] && TARGETS="$TARGETS;$P1"
    [ $((M & 2)) -ne 0 ] && TARGETS="$TARGETS;$P2"
    [ $((M & 4)) -ne 0 ] && TARGETS="$TARGETS;$P3"
    TARGETS=${TARGETS#;} ;;
  *) echo "ABORT: unknown set $SET"; exit 2 ;;
esac
[ -x "$BL" ] && [ -f "$CLEVR_GEN/render_images.py" ] && [ -f "$S" ] || { echo "ABORT: renderer or snapshot missing on $(hostname -s)"; exit 3; }
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full"; exit 3; }
THREADS=$(nproc)
echo "[$(date -u +%H:%M:%S)] job=${SLURM_JOB_ID:-none} node=$(hostname -s) threads=$THREADS set=$SET targets='$TARGETS' start=$START count=$COUNT snap=$SNAP"
# Jobs starting together can race on this shared disk, so the folder is retried until it is visible.
for _ in 1 2 3 4 5 6; do mkdir -p "$OUT/$SET" 2>/dev/null; [ -d "$OUT/$SET" ] && break; sleep 5; done
[ -d "$OUT/$SET" ] || { echo "ABORT: could not create $OUT/$SET"; exit 3; }
"$BL" --background --python-exit-code 7 --threads "$THREADS" --python "$S" -- --name "$SET" --targets "$TARGETS" --seed-base "$BASE" \
  --start "$START" --count "$COUNT" --out "$OUT" 2>&1 | grep --line-buffered -E 'M7RENDER|Traceback|Error:|error:'
rc=${PIPESTATUS[0]}
MISSING=0
for ((i = START; i < START + COUNT; i++)); do
  [ -f "$OUT/$SET/scenes/$(printf '%s_%06d.json' "$SET" "$i")" ] || MISSING=$((MISSING + 1))
done
echo "[$(date -u +%H:%M:%S)] DONE set=$SET exit=$rc asked=$COUNT missing=$MISSING"
[ "$rc" -eq 0 ] && [ "$MISSING" -eq 0 ] || exit 9
exit 0
