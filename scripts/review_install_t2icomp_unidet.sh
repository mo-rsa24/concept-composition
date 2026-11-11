#!/bin/bash
# T2I-CompBench's spatial (UniDet) and complex (3-in-1: BLIP-VQA, UniDet, CLIP) scorers, added to the
# t2icomp env that scripts/review_install_t2icomp.sh made for BLIP-VQA. Pins from the repo's own
# requirements.txt at commit 4aa4042: detectron2 at 5aeb252, accelerate 0.17.0, openai CLIP, fvcore,
# iopath, hydra-core, omegaconf, pycocotools; weights per its README, section 2.
# No CUDA 11.8 toolkit exists on the nodes (only 12.9 and 13.3), so detectron2 builds its C++ ops for the
# CPU only (FORCE_CUDA=0); UniDet's ROI pooling and NMS run through torchvision's CUDA ops.
set -euxo pipefail
export CONDA_PKGS_DIRS=/datasets/mmolefe/conda-pkgs
ENV=/datasets/mmolefe/envs/t2icomp; R=/datasets/mmolefe/repos/T2I-CompBench
PIP="$ENV/bin/python -m pip"
$PIP install accelerate==0.17.0 fvcore==0.1.5.post20221221 iopath==0.1.9 hydra-core==1.3.2 omegaconf==2.3.0 \
  pycocotools==2.0.6 ftfy==6.1.1 regex==2023.6.3 yacs==0.1.8 cloudpickle==2.2.1 matplotlib==3.7.1 \
  tabulate==0.9.0 termcolor==2.3.0 black==22.3.0 pydot==1.4.2 tensorboard==2.13.0 gdown
$PIP install "git+https://github.com/openai/CLIP.git"
FORCE_CUDA=0 $PIP install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13"
mkdir -p $R/UniDet_eval/experts/expert_weights; cd $R/UniDet_eval/experts/expert_weights
[ -s Unified_learned_OCIM_RS200_6x+2x.pth ] || wget -q -O Unified_learned_OCIM_RS200_6x+2x.pth \
  "https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth"
cd $R/UniDet_eval
$ENV/bin/python -c "import detectron2, clip, accelerate; from experts.model_bank import load_expert_model; print('UNIDET_OK', detectron2.__version__, accelerate.__version__)"
