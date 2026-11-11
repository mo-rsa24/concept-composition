#!/bin/bash
# Second approach: mmcv-full 1.7.1 by its cp39 wheel URL (pip skipped it from the find-links page),
# then mmdetection 2.x without build isolation so its build sees the env's torch.
set -euxo pipefail
export CONDA_PKGS_DIRS=/datasets/mmolefe/conda-pkgs
source /home-mscluster/mmolefe/miniforge3/etc/profile.d/conda.sh
ENV=/datasets/mmolefe/envs/geneval; R=/datasets/mmolefe/repos
conda activate $ENV
pip install https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/mmcv_full-1.7.1-cp39-cp39-manylinux1_x86_64.whl
cd $R/mmdetection; git checkout -q 2.x; pip install --no-build-isolation -e .
pip install open-clip-torch==2.20.0 clip-benchmark==1.4.0 "numpy<1.24" pandas tqdm opencv-python==4.6.0.66
bash $R/geneval/evaluation/download_models.sh $R/geneval_models
python -c "import torch, mmcv, mmdet, open_clip; from mmcv.ops import nms; print('GENEVAL_OK', torch.__version__, mmcv.__version__, mmdet.__version__, open_clip.__version__)"
