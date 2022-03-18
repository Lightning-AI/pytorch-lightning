# Copyright The PyTorch Lightning team.
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

ARG CUDA_VERSION=11.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=1.8

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Prague \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5;8.0" \
    MKL_THREADING_LAYER=GNU \
    # MAKEFLAGS="-j$(nproc)"
    MAKEFLAGS="-j2"

RUN apt-get update -qq --fix-missing  && \
    apt-get install -y --no-install-recommends \
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
    && \

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

COPY ./requirements.txt requirements.txt
COPY ./requirements/ ./requirements/
COPY ./.actions/assistant.py assistant.py

ENV PYTHONPATH=/usr/lib/python${PYTHON_VERSION}/site-packages

RUN \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \

    pip install -q fire && \
    # Disable cache \
    CUDA_VERSION_MM=$(python -c "print(''.join('$CUDA_VERSION'.split('.')[:2]))") && \
    export BAGUA_CUDA_VERSION=$CUDA_VERSION_MM && \
    pip config set global.cache-dir false && \
    # set particular PyTorch version
    python ./requirements/adjust-versions.py requirements.txt ${PYTORCH_VERSION} && \
    python ./requirements/adjust-versions.py requirements/extra.txt ${PYTORCH_VERSION} && \
    python ./requirements/adjust-versions.py requirements/examples.txt ${PYTORCH_VERSION} && \
    python -c "print(' '.join([ln for ln in open('requirements/extra.txt').readlines() if 'horovod' in ln]))" > ./requirements/horovod.txt && \
    python assistant.py requirements_prune_pkgs requirements/examples.txt "horovod" && \
    # Install all requirements \
    pip install -r requirements/devel.txt --no-cache-dir --find-links https://download.pytorch.org/whl/cu${CUDA_VERSION_MM}/torch_stable.html && \
    rm -rf requirements.* && \
    rm assistant.py

RUN \
   apt-get purge -y cmake && \
   wget -q https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz && \
   tar -zxvf cmake-3.20.2.tar.gz && \
   cd cmake-3.20.2 && \
   ./bootstrap -- -DCMAKE_USE_OPENSSL=OFF && \
   make && \
   make install && \
   cmake  --version

ENV \
    HOROVOD_CUDA_HOME=$CUDA_TOOLKIT_ROOT_DIR \
    HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITH_GLOO=1 \
    HOROVOD_WITHOUT_MPI=1

RUN \
   HOROVOD_BUILD_CUDA_CC_LIST=${TORCH_CUDA_ARCH_LIST//";"/","} && \
   export HOROVOD_BUILD_CUDA_CC_LIST=${HOROVOD_BUILD_CUDA_CC_LIST//"."/""} && \
   cat ./requirements/horovod.txt && \
   cmake --version && \
   pip install --no-cache-dir -r ./requirements/horovod.txt && \
   PYTHONUNBUFFERED=1 horovodrun --check-build && \
   pip uninstall -y horovod && \
   pip install --no-cache-dir -r ./requirements/horovod.txt && \
   PYTHONUNBUFFERED=1 horovodrun --check-build && \
   python -c "from horovod.torch import nccl_built; nccl_built()" && \
   rm -rf requirements/

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__"
