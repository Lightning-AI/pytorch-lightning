# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.7.1


FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.0
ARG MAX_ALLOWED_NCCL=2.16.2

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER="GNU" \
    # MAKEFLAGS="-j$(nproc)"
    MAKEFLAGS="-j2"

RUN \
    # TODO: Remove the manual key installation once the base image is updated.
    # https://github.com/NVIDIA/nvidia-docker/issues/1631
    # https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1264715214
    apt-get update && apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    mkdir -p /etc/apt/keyrings/ && mv 3bf863cc.pub /etc/apt/keyrings/ && \
    echo "deb [signed-by=/etc/apt/keyrings/3bf863cc.pub] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" /etc/apt/sources.list.d/cuda.list && \
    apt-get update -qq --fix-missing && \
    NCCL_VER=$(dpkg -s libnccl2 | grep '^Version:' | awk -F ' ' '{print $2}' | awk -F '-' '{print $1}' | grep -ve '^\s*$') && \
    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
    TO_INSTALL_NCCL=$(echo -e "$MAX_ALLOWED_NCCL\n$NCCL_VER" | sort -V  | head -n1)-1+cuda${CUDA_VERSION_MM} && \
    apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        build-essential \
        pkg-config \
        cmake \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        software-properties-common \
        libopenmpi-dev \
        openmpi-bin \
        ssh \
        ninja-build \
        libnccl2=$TO_INSTALL_NCCL \
        libnccl-dev=$TO_INSTALL_NCCL && \
    # Install python
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/pytorch/ requirements/pytorch/
COPY requirements/_integrations/ requirements/_integrations/

ENV PYTHONPATH="/usr/lib/python${PYTHON_VERSION}/site-packages"

RUN \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    # Disable cache \
    pip config set global.cache-dir false && \
    # set particular PyTorch version \
    for fpath in `ls requirements/**/*.txt`; do \
      python ./requirements/pytorch/adjust-versions.py $fpath ${PYTORCH_VERSION}; \
    done && \
    CUDA_VERSION_MM=${CUDA_VERSION%.*} && \
    pip install \
      -r requirements/pytorch/devel.txt \
      -r requirements/pytorch/strategies.txt \
      --find-links "https://download.pytorch.org/whl/cu${CUDA_VERSION_MM//'.'/''}/torch_stable.html"

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__" && \
    python requirements/pytorch/check-avail-extras.py && \
    rm -rf requirements/
