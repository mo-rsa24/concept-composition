#!/bin/bash
# paper2_vlm_env_build.sh: a separate environment for the vision-language judge (Qwen3-VL-8B) on /datasets.
#
#   bash 2-classifier-guidance/scripts/paper2/paper2_vlm_env_build.sh
#
# The owner said yes (progress board question 120): if co3 cannot load Qwen3-VL-8B, this environment serves
# the vision-language judge only; the other three judges, the dial trainer and CO3 stay on co3_bw. Runs at
# the owner's launch, with the weight fetch, on the held Blackwell node. Builds into
# /datasets/mmolefe/envs/paper2-vlm (P2_VLM_ENV overrides), never /home-mscluster, with the conda and pip
# caches on /datasets too, since their defaults sit in the home folder (environment/storage.md, the disk
# guard rule). Stops if the folder already exists. Prints each version and the folder's size at the end.
#
# Torch is built for CUDA 12.8, as a Blackwell card needs (co3's cu118 torch runs on the CPU there with no
# error). Every version is pinned; each exists on PyPI or the PyTorch cu128 index (checked 2026-11-26 with
# `pip index versions`). transformers 4.57.6 is the last 4.x release and contains Qwen3-VL; it needs
# huggingface_hub below 1.0 and tokenizers 0.22.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paper2_guards.sh"
p2_log paper2-vlm-env-build
PREFIX="$P2_VLM_ENV"
CONDA=/home-mscluster/mmolefe/miniforge3/bin/conda
PYTHON_VERSION=3.11
TORCH=2.11.0
TORCHVISION=0.26.0
TORCH_INDEX=https://download.pytorch.org/whl/cu128
PINS=(transformers==4.57.6 huggingface_hub==0.36.2 tokenizers==0.22.2 accelerate==1.12.0 safetensors==0.7.0
      pillow==12.1.1 numpy==2.3.5)

case "$PREFIX" in /datasets/*) ;; *) p2_abort 7 "the environment must go under /datasets, not $PREFIX";; esac
[ -e "$PREFIX" ] && p2_abort 7 "$PREFIX already exists; remove it by hand first if a rebuild is wanted"
[ -x "$CONDA" ] || p2_abort 5 "no conda at $CONDA"
p2_disk_guard "$(dirname "$PREFIX")"
export CONDA_PKGS_DIRS=/datasets/mmolefe/.cache/conda-pkgs PIP_CACHE_DIR=/datasets/mmolefe/.cache/pip
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
p2_disk_guard "$CONDA_PKGS_DIRS"
echo "conda packages cache: $CONDA_PKGS_DIRS; pip cache: $PIP_CACHE_DIR"

"$CONDA" create --yes --prefix "$PREFIX" "python=$PYTHON_VERSION" pip || p2_abort 1 "conda create failed"
"$PREFIX/bin/python" -m pip install --index-url "$TORCH_INDEX" "torch==$TORCH" "torchvision==$TORCHVISION" \
  || p2_abort 1 "torch $TORCH for cu128 did not install"
"$PREFIX/bin/python" -m pip install "${PINS[@]}" || p2_abort 1 "the pinned packages did not install"

"$PREFIX/bin/python" - <<'EOF'
import importlib.metadata as md, sys, torch
print("python", sys.version.split()[0])
for p in ("torch", "torchvision", "transformers", "huggingface_hub", "tokenizers", "accelerate", "safetensors",
          "pillow", "numpy"):
    print(p, md.version(p))
print("torch cuda build", torch.version.cuda)
from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
print("transformers has Qwen3-VL")
EOF
[ $? -eq 0 ] || p2_abort 1 "the environment does not import"
echo "size: $(du -sh "$PREFIX" | cut -f1) at $PREFIX"
echo "next: bash 2-classifier-guidance/scripts/paper2/paper2_vlm_env_check.sh"
