#!/bin/bash
# GenEval env per djghosh13/geneval environment.yml pins (python 3.9, torch 1.12.1 cu113, mmcv-full 1.7.1, mmdet 2.x)
set -euxo pipefail
source /home-mscluster/mmolefe/miniforge3/etc/profile.d/conda.sh
ENV=/datasets/mmolefe/envs/geneval; R=/datasets/mmolefe/repos
conda create -y -p $ENV python=3.9
conda activate $ENV
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
cd $R; [ -d mmdetection ] || git clone -q https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout -q 2.x; pip install -e .
pip install open-clip-torch==2.20.0 clip-benchmark==1.4.0 "numpy<1.24" pandas tqdm opencv-python==4.6.0.66
bash $R/geneval/evaluation/download_models.sh $R/geneval_models
python -c "import torch, mmcv, mmdet, open_clip; print('GENEVAL_OK', torch.__version__, mmcv.__version__, mmdet.__version__, open_clip.__version__)"
