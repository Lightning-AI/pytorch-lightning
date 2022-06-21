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

ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.3.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=1.9

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

RUN \
    # TODO: Remove the manual key installation once the base image is updated.
    # https://github.com/NVIDIA/nvidia-docker/issues/1631
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update -qq --fix-missing && \
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
        openmpi-bin \
        ssh \
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
    pip config set global.cache-dir false && \
    # set particular PyTorch version
    python ./requirements/adjust-versions.py requirements/base.txt ${PYTORCH_VERSION} && \
    python ./requirements/adjust-versions.py requirements/extra.txt ${PYTORCH_VERSION} && \
    python ./requirements/adjust-versions.py requirements/examples.txt ${PYTORCH_VERSION} && \
    # Install all requirements \
    pip install -r requirements/devel.txt --no-cache-dir --find-links https://download.pytorch.org/whl/cu${CUDA_VERSION_MM}/torch_stable.html && \
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
    HOROVOD_WITH_MPI=1

RUN \
    # CUDA 10.2 doesn't support ampere architecture (8.0).
    if [[ "$CUDA_VERSION" < "11.0" ]]; then export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST//";8.0"/}; echo $TORCH_CUDA_ARCH_LIST; fi && \
    HOROVOD_BUILD_CUDA_CC_LIST=${TORCH_CUDA_ARCH_LIST//";"/","} && \
    export HOROVOD_BUILD_CUDA_CC_LIST=${HOROVOD_BUILD_CUDA_CC_LIST//"."/""} && \
    echo $HOROVOD_BUILD_CUDA_CC_LIST && \
    cmake --version && \
    pip install --no-cache-dir -r ./requirements/strategies.txt && \
    horovodrun --check-build

RUN \
    CUDA_VERSION_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])") && \
    py_ver=$(python -c "print(int('$PYTHON_VERSION'.split('.') >= '3.9'.split('.')))") && \
    # install DALI, needed for examples
    # todo: waiting for 1.4 - https://github.com/NVIDIA/DALI/issues/3144#issuecomment-877386691
    if [ $py_ver -eq "0" ]; then \
        pip install --extra-index-url https://developer.download.nvidia.com/compute/redist "nvidia-dali-cuda${CUDA_VERSION_MAJOR}0>1.0" ; \
        python -c 'from nvidia.dali.pipeline import Pipeline' ; \
    fi

RUN \
    # CUDA 10.2 doesn't support ampere architecture (8.0).
    if [[ "$CUDA_VERSION" < "11.0" ]]; then export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST//";8.0"/}; echo $TORCH_CUDA_ARCH_LIST; fi && \
    # install NVIDIA apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex/archive/refs/heads/master.zip && \
    python -c "from apex import amp"

RUN \
    # install Bagua
    CUDA_VERSION_MM=$(python -c "print(''.join('$CUDA_VERSION'.split('.')[:2]))") && \
    pip install "bagua-cuda$CUDA_VERSION_MM==0.9.0" && \
    python -c "import bagua_core; bagua_core.install_deps()" && \
    python -c "import bagua; print(bagua.__version__)"

COPY requirements/check-avail-extras.py check-avail-extras.py
COPY requirements/check-avail-strategies.py check-avail-strategies.py

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__" && \
    python requirements/check-avail-extras.py && \
    python requirements/check-avail-strategies.py && \
    rm -rf requirements/
