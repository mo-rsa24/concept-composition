# The cluster's co3 environment as a RunPod image: Python 3.10, PyTorch 2.5.1 for CUDA 11.8, and every other
# package at the exact version `pip list` reports for /home-mscluster/mmolefe/miniforge3/envs/co3 (requirements.txt,
# built from co3-freeze-raw.txt). Code and data are not baked in: they arrive by rsync when a pod starts.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/workspace/hf
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv git openssh-server rsync curl ca-certificates libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && rm -rf /var/lib/apt/lists/*

# PyTorch and xformers from the CUDA 11.8 index, exactly as in co3.
RUN python -m pip install --upgrade pip==26.0.1 \
    && python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 xformers==0.0.28.post3 \
        --index-url https://download.pytorch.org/whl/cu118

# Everything else pinned and installed without dependency resolution, so nothing can upgrade PyTorch.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-deps -r /tmp/requirements.txt \
    && python -c "import torch, diffusers, transformers; assert torch.__version__.startswith('2.5.1'), torch.__version__; print(torch.__version__, diffusers.__version__, transformers.__version__)"

COPY start.sh /start.sh
RUN chmod +x /start.sh
WORKDIR /workspace
CMD ["/start.sh"]
