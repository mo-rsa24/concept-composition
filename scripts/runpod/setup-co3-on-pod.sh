#!/bin/bash
# Build the cluster's co3 environment on a fresh RunPod pod started from any Ubuntu 22.04 image
# (RunPod's PyTorch templates are Ubuntu 22.04). Uses the system Python 3.10, as co3 does, so no
# custom image has to be pulled. Every package is pinned to the version `pip list` reports for co3,
# installed without dependency resolution so nothing can upgrade PyTorch.
#
# Usage on the pod:  bash setup-co3-on-pod.sh /workspace/requirements.txt
# Then:              source /workspace/co3/bin/activate
set -euo pipefail
REQ="${1:-/workspace/requirements.txt}"
export DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
apt-get update -qq
apt-get install -y -qq libgl1 libglib2.0-0 rsync curl >/dev/null
# RunPod's current PyTorch templates are Ubuntu 24.04, whose system Python is 3.12, so uv supplies
# a standalone Python 3.10, the version co3 uses.
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 /workspace/co3
. /workspace/co3/bin/activate
uv pip install -q torch==2.5.1+cu118 torchvision==0.20.1+cu118 xformers==0.0.28.post3 \
    --index-url https://download.pytorch.org/whl/cu118
uv pip install -q --no-deps -r "$REQ"
python -c "import sys, torch, diffusers, transformers; print('python', sys.version.split()[0], 'torch', torch.__version__, 'diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
