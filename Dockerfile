FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    ninja-build \
    cmake \
    pkg-config \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv

ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=9.0
ENV PYTHONPATH=/workspace/xllm:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}
ENV MAX_JOBS=8
ENV NVCC_THREADS=4
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV XLLM_REPO_ROOT=/workspace/xllm

WORKDIR /workspace/xllm

CMD ["bash"]
