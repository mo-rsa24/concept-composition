#!/bin/bash
# paper2_fetch_followon.sh: the follow-on judges' weights (step 0 failed; the owner chose to test DreamSim and BioCLIP).
#
#   bash 2-classifier-guidance/scripts/paper2/paper2_fetch_followon.sh            # fetch, then check offline
#   bash 2-classifier-guidance/scripts/paper2/paper2_fetch_followon.sh --check-only
#
# CPU work, on the held node, from the cluster checkout, behind the same guards as paper2_fetch_judges.sh. BioCLIP
# (imageomics/bioclip) goes into HF_HOME on /datasets through paper2_fetch_weights.py --set followon, and must build
# through open_clip offline. DreamSim's weights come through the dreamsim package, into
# /datasets/mmolefe/poe_repair_min/paper2/weights/dreamsim (its cache_dir, with TORCH_HOME and HF_HOME pointed there as
# well, so nothing lands in the home folder); then the same constructor runs again with HF_HUB_OFFLINE=1 and embeds one
# grey picture. Whether co3_bw has the dreamsim and open_clip packages is not recorded in environment/: the import is
# tried first and a missing package stops the script with its name, before anything is fetched.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paper2_guards.sh"
cd "$P2_REPO" || exit 7
p2_log paper2-fetch-followon
p2_disk_guard "$P2_HF_HOME"
p2_disk_guard "$P2_ROOT/weights"
p2_python
export HF_HOME="$P2_HF_HOME"
DS="$P2_ROOT/weights/dreamsim"
mkdir -p "$DS/torch" outputs/paper2/paper2-fetch-weights
OUT=outputs/paper2/paper2-fetch-weights/sizes-followon.txt
for mod in open_clip dreamsim; do
  "$PY" -c "import $mod" 2>/dev/null || { echo "FAIL: $PY cannot import $mod; install it in that environment first"; exit 4; }
done
HOME_CACHES=("$HOME/.cache/huggingface" "$HOME/.cache/torch" "$HOME/.cache/pip")
home_bytes() { du -scb "${HOME_CACHES[@]}" 2>/dev/null | tail -1 | cut -f1; }
BEFORE=$(home_bytes)
"$PY" "$P2_SCRIPTS/paper2_fetch_weights.py" --set followon --out "$OUT" "$@"
rc=$?
CHECK='import os,sys
from dreamsim import dreamsim
from PIL import Image
m, pre = dreamsim(pretrained=True, device="cpu", cache_dir=sys.argv[1])
e = m.embed(pre(Image.new("RGB", (224, 224), (128, 128, 128))))
print("dreamsim builds and embeds offline:" if os.environ.get("HF_HUB_OFFLINE") else "dreamsim fetched and embeds:", tuple(e.shape))'
if [ "${1:-}" != "--check-only" ]; then
  TORCH_HOME="$DS/torch" "$PY" -c "$CHECK" "$DS" 2>&1 | tail -3 | tee -a "$OUT"
fi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$DS/torch" "$PY" -c "$CHECK" "$DS" 2>&1 | tail -3 | tee -a "$OUT"
ds=${PIPESTATUS[0]}
echo "dreamsim folder: $(du -sh "$DS" | cut -f1)" | tee -a "$OUT"
GROWTH=$(( $(home_bytes) - ${BEFORE:-0} ))
echo "home caches (${HOME_CACHES[*]}) grew by $GROWTH bytes" | tee -a "$OUT"
cp "$OUT" "$P2_ROOT/paper2-fetch-weights/sizes-followon.txt"
if [ "$GROWTH" -gt 1000000000 ]; then echo "FAIL: the home folder's caches grew by more than 1 GB"; exit 3; fi
[ $rc -eq 0 ] && [ "$ds" -eq 0 ] && echo "PASS: BioCLIP and DreamSim load offline" || { echo "FAIL: BioCLIP exit $rc, DreamSim exit $ds"; exit 1; }
