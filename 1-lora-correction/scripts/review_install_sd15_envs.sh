#!/bin/bash
# The two environments for compute job 10 (SD 1.5 baselines for Table 4) and the SynGen parse of the SDXL port.
# Plan: 1-lora-correction/artifacts/drips/answer-iclr-reviews/routes/plan-sd15-baselines-and-syngen-sdxl.md, "The two new environments".
#
#   sd15-aae  Attend-and-Excite (composition/aae, = yuval-alaluf/Attend-and-Excite@163efdf) and Divide-and-Bind
#             (boschresearch/Divide-and-Bind@644dfda), whose requirements.txt both pin torch 1.12 / diffusers 0.12.1 /
#             transformers 4.26.0. huggingface_hub is pinned because diffusers 0.12.1 imports cached_download, which
#             huggingface_hub 0.26 removed. safetensors is added because the SD 1.5 snapshot holds only .safetensors
#             weights, and neither diffusers 0.12.1 nor transformers 4.26.0 can read them without it.
#   syngen    RoyiRa/Linguistic-Binding-in-Diffusion-Models@50b8eab's environment.yaml as released, plus en_core_web_trf
#             3.6.1 from its release wheel (the parser its README names).
#
# Runs on a GPU node that reaches PyPI (the import line loads each pipeline and checks CUDA). Shared-device form:
#   timeout 30 ssh mscluster107 'GPU=0 nohup bash <repo>/scripts/review_install_sd15_envs.sh > <log> 2>&1 < /dev/null & disown'
# ENVS="sd15-aae syngen" by default; ENVS=syngen installs one. An environment folder that exists is not recreated.
set -euxo pipefail
source /home-mscluster/mmolefe/miniforge3/etc/profile.d/conda.sh
ROOT=/datasets/mmolefe/envs
REPOS=/datasets/mmolefe/repos
REPO="${REPO:-/home-mscluster/mmolefe/Playground/PhD/poe_repair_min}"
ENVS="${ENVS:-sd15-aae syngen}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
echo "=== install $ENVS  node $(hostname -s)  device $CUDA_VISIBLE_DEVICES  pid $$  $(date -Is)"
USE=$(df --output=pcent "$ROOT" | tail -1 | tr -dc '0-9')
[ -n "$USE" ] && [ "$USE" -lt 90 ] || { echo "ABORT: $ROOT is ${USE:-unknown}% full" >&2; exit 3; }

for E in $ENVS; do
  P=$ROOT/$E
  case "$E" in
    sd15-aae)
      if [ ! -x "$P/bin/python" ]; then
        conda create -y -p "$P" python=3.9
        "$P/bin/pip" install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
        "$P/bin/pip" install diffusers==0.12.1 transformers==4.26.0 huggingface_hub==0.13.4 accelerate==0.16.0 \
          numpy==1.23.5 pyrallis==0.3.1 opencv-python-headless==4.7.0.72 einops==0.6.0 safetensors==0.3.1 \
          ipython matplotlib ftfy scipy
      fi
      # The two repositories both ship a top-level package named utils, so each is imported in its own process.
      "$P/bin/python" - "$REPO/composition/aae" <<'EOF'
import sys; sys.path.insert(0, sys.argv[1])
import torch, diffusers, transformers, huggingface_hub
from pipeline_attend_and_excite import AttendAndExcitePipeline
from utils.ptp_utils import AttentionStore, register_attention_control
assert torch.cuda.is_available(), "torch sees no CUDA"
print("SD15_AAE_OK attend-and-excite", "torch", torch.__version__, "diffusers", diffusers.__version__,
      "transformers", transformers.__version__, "hub", huggingface_hub.__version__, torch.cuda.get_device_name(0))
EOF
      "$P/bin/python" - "$REPOS/Divide-and-Bind" <<'EOF'
import sys; sys.path.insert(0, sys.argv[1])
import torch, diffusers
from divide_and_bind.pipeline_divide_and_bind import DivideAndBindPipeline
from divide_and_bind.config import RunConfig
from utils.ptp_utils import AttentionStore, register_attention_control
from utils.test_utils import default_param
assert torch.cuda.is_available(), "torch sees no CUDA"
print("SD15_AAE_OK divide-and-bind", "torch", torch.__version__, "diffusers", diffusers.__version__, default_param("tv_bind"))
EOF
      ;;
    syngen)
      if [ ! -x "$P/bin/python" ]; then
        conda create -y -p "$P" python=3.11.2 pip=23.0.1
        "$P/bin/pip" install diffusers==0.21.1 numpy==1.23.5 spacy==3.6.1 tqdm==4.65.0 transformers==4.30.2 torch==2.0.1 \
          accelerate==0.23.0 huggingface_hub==0.23.2
        "$P/bin/pip" install "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.6.1/en_core_web_trf-3.6.1-py3-none-any.whl"
      fi
      "$P/bin/python" - "$REPOS/Linguistic-Binding-in-Diffusion-Models" <<'EOF'
import sys; sys.path.insert(0, sys.argv[1])
import torch, diffusers, transformers, spacy
from syngen_diffusion_pipeline import SynGenDiffusionPipeline
nlp = spacy.load("en_core_web_trf")
doc = nlp("a pink crown and a purple bow")
assert torch.cuda.is_available(), "torch sees no CUDA"
print("SYNGEN_OK", "torch", torch.__version__, "diffusers", diffusers.__version__, "transformers", transformers.__version__,
      "spacy", spacy.__version__, [(t.text, t.dep_) for t in doc], torch.cuda.get_device_name(0))
EOF
      ;;
    *) echo "ABORT: unknown environment $E" >&2; exit 2 ;;
  esac
done
echo "=== install done $(date -Is)"
