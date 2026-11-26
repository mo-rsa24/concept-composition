#!/bin/bash
# paper2_upload_joint.sh: setup plan task 7, the joint prompt's 192 pictures from the laptop to /datasets.
#
#   bash 2-classifier-guidance/scripts/paper2/paper2_upload_joint.sh --go
#
# Runs on the LAPTOP, from the laptop checkout. Waits for the owner's route (progress board question 119);
# the route written here is the one that question proposes: rsync through mscluster84 in the push
# direction, the reverse of `scripts/cluster.sh pull`. Without --go it prints what it would do and stops.
# Checks the remote disk before copying, copies outputs/loop-6/m14/blind/pictures/ (193 files: 192
# pictures and prompts.csv), then counts the pictures on the cluster and checks each opens at 1024 by 1024.
# Passes when 192 pictures open there at that size. After it passes, record the route in
# environment/storage.md (this script does not edit it).
set -uo pipefail
NODE="${CLUSTER_NODE:-mscluster84}"
SRC=outputs/loop-6/m14/blind/pictures/
DEST=/datasets/mmolefe/poe_repair_min/paper2/paper2-upload-joint/
PY=/home-mscluster/mmolefe/miniforge3/envs/co3/bin/python
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 7

N=$(find "$SRC" -maxdepth 1 -type f | wc -l)
NPNG=$(find "$SRC" -maxdepth 1 -type f -name '*.png' | wc -l)
echo "laptop: $SRC holds $N files, $NPNG pictures"
[ "$NPNG" -eq 192 ] || { echo "ABORT: expected 192 pictures, found $NPNG"; exit 7; }
echo "route: rsync -az $SRC $NODE:$DEST"
[ "${1:-}" = "--go" ] || { echo "dry: nothing copied; run with --go once the owner has named the route (question 119)"; exit 0; }

ssh "$NODE" "mkdir -p $DEST && df --output=pcent $DEST | tail -1" | tr -dc '0-9' | {
  read -r USE; [ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $DEST is ${USE:-unknown}% full"; exit 3; }
  echo "disk: $DEST is ${USE}% full"; } || exit 3
rsync -az --info=progress2 "$SRC" "$NODE:$DEST" || { echo "ABORT: rsync failed"; exit 1; }
ssh "$NODE" "$PY - <<'EOF'
from pathlib import Path
from PIL import Image
d = Path('$DEST')
pics = sorted(d.glob('*.png'))
bad = [p.name for p in pics if Image.open(p).size != (1024, 1024)]
print(f'cluster: {len(pics)} pictures in {d}, {len(pics) - len(bad)} open at 1024 by 1024')
raise SystemExit(0 if len(pics) == 192 and not bad else 1)
EOF" && echo "PASS: 192 pictures at 1024 by 1024 on /datasets" || { echo "FAIL: count or size check"; exit 1; }
