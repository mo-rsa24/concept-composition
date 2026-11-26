#!/bin/bash
# paper2_co3_load.sh: setup plan task 8, CO3 builds from the fetched SDXL weights, offline, no picture drawn.
#
#   bash 2-classifier-guidance/scripts/paper2/paper2_co3_load.sh
#
# Only after the owner's yes and after task 6. Run from the cluster checkout on the held Blackwell node.
# CO3 turns on xformers attention as it loads, which needs a visible card, so the build uses device 0
# (about 10 GB; Paper 1's process may share the card). Stops before loading if the SDXL snapshot, the
# fp16-fix VAE CO3 loads beside it, or the sentence parser's English model is missing, so a skipped task 6
# cannot cost a run. This is where co3_common.py (method_search/, restored 2026-11-26) is first proven
# to load CO3. Writes outputs/paper2/paper2-co3-load/load.txt in the checkout and a copy beside the log.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paper2_guards.sh"
cd "$P2_REPO" || exit 7
p2_log paper2-co3-load
p2_disk_guard "$P2_ROOT/paper2-co3-load"
p2_python
p2_hf_env
for repo in models--stabilityai--stable-diffusion-xl-base-1.0 models--madebyollin--sdxl-vae-fp16-fix; do
  ls -d "$HF_HOME/hub/$repo/snapshots/"*/ >/dev/null 2>&1 \
    || p2_abort 7 "$repo is not in $HF_HOME: run task 6 (paper2_fetch_weights.sh) first"
done
ls "$HF_HOME/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/"*/unet/*fp16.safetensors >/dev/null 2>&1 \
  || p2_abort 7 "SDXL's fp16 U-Net is not in $HF_HOME: run task 6 first"
STANZA_DIR="${STANZA_RESOURCES_DIR:-$HOME/stanza_resources}"
[ -d "$STANZA_DIR/en" ] || p2_abort 7 "CO3's sentence parser model is not at $STANZA_DIR/en (stanza loads it with download_method=None)"
echo "stanza: $STANZA_DIR/en"
p2_gpu_guard 12000
export CUDA_VISIBLE_DEVICES=0
p2_cuda_probe "$PY"
p2_co3_path
"$PY" "$P2_SCRIPTS/paper2_co3_load.py" --out outputs/paper2/paper2-co3-load/load.txt
rc=$?
[ -f outputs/paper2/paper2-co3-load/load.txt ] && cp outputs/paper2/paper2-co3-load/load.txt "$P2_ROOT/paper2-co3-load/load.txt"
[ $rc -eq 0 ] && echo "PASS: CO3 built offline" || echo "FAIL: CO3 did not build (exit $rc); the log names the missing file or package"
exit $rc
