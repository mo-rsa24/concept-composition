#!/bin/bash
# T2I-CompBench BLIP-VQA env per Karine-Huang/T2I-CompBench requirements.txt pins
set -euxo pipefail
source /home-mscluster/mmolefe/miniforge3/etc/profile.d/conda.sh
ENV=/datasets/mmolefe/envs/t2icomp
conda create -y -p $ENV python=3.9
conda activate $ENV
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.2 timm==0.4.12 fairscale==0.4.4 spacy==3.5.3 numpy==1.25.0 Pillow==9.5.0 tqdm ruamel.yaml pyyaml
pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl"
python -c "import torch, transformers, timm, spacy; spacy.load('en_core_web_sm'); print('T2I_OK', torch.__version__, transformers.__version__, timm.__version__, spacy.__version__)"
