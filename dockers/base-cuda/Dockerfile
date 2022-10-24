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
ARG PYTORCH_VERSION=1.12

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
    NCCL_VER=$(dpkg -s libnccl2 | grep '^Version:' | awk -F ' ' '{print $2}' | awk -F '-' '{print $1}' | grep -ve '^\s*$') && \
    CUDA_VERSION_MM="${CUDA_VERSION%.*}" && \
    MAX_ALLOWED_NCCL=2.11.4 && \
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

COPY ./requirements/pytorch/ ./requirements/pytorch/
COPY ./.actions/assistant.py assistant.py

ENV PYTHONPATH=/usr/lib/python${PYTHON_VERSION}/site-packages

RUN \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    pip install -q fire && \
    # Disable cache \
    export CUDA_VERSION_MM=$(python -c "print(''.join('$CUDA_VERSION'.split('.')[:2]))") && \
    pip config set global.cache-dir false && \
    # set particular PyTorch version
    python ./requirements/pytorch/adjust-versions.py requirements/pytorch/base.txt ${PYTORCH_VERSION} && \
    python ./requirements/pytorch/adjust-versions.py requirements/pytorch/extra.txt ${PYTORCH_VERSION} && \
    python ./requirements/pytorch/adjust-versions.py requirements/pytorch/examples.txt ${PYTORCH_VERSION} && \

    # Install base requirements \
    pip install -r requirements/pytorch/base.txt --no-cache-dir --find-links https://download.pytorch.org/whl/cu${CUDA_VERSION_MM}/torch_stable.html && \
    rm assistant.py

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
    pip install --no-cache-dir horovod && \
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
    CUDA_VERSION_BAGUA=$(python -c "print([ver for ver in [116,113,111,102] if $CUDA_VERSION_MM >= ver][0])") && \
    pip install "bagua-cuda$CUDA_VERSION_BAGUA" && \
    if [[ "$CUDA_VERSION_MM" = "$CUDA_VERSION_BAGUA" ]]; then python -c "import bagua_core; bagua_core.install_deps()"; fi && \
    python -c "import bagua; print(bagua.__version__)"

RUN \
    # install ColossalAI
    SHOULD_INSTALL_COLOSSAL=$(python -c "import torch; print(1 if int(torch.__version__.split('.')[1]) > 9 else 0)") && \
    if [[ "$SHOULD_INSTALL_COLOSSAL" = "1" ]]; then \
        PYTORCH_VERSION_COLOSSALAI=$(python -c "import torch; print(torch.__version__.split('+')[0][:4])") ; \
        CUDA_VERSION_MM_COLOSSALAI=$(python -c "import torch ; print(''.join(map(str, torch.version.cuda)))") ; \
        CUDA_VERSION_COLOSSALAI=$(python -c "print([ver for ver in [11.3, 11.1] if $CUDA_VERSION_MM_COLOSSALAI >= ver][0])") ; \
        pip install "colossalai==0.1.10+torch${PYTORCH_VERSION_COLOSSALAI}cu${CUDA_VERSION_COLOSSALAI}" --find-links https://release.colossalai.org ; \
        python -c "import colossalai; print(colossalai.__version__)" ; \
    fi

RUN \
    # install rest of strategies
    # remove colossalai from requirements since they are installed separately
    SHOULD_INSTALL_COLOSSAL=$(python -c "import torch; print(1 if int(torch.__version__.split('.')[1]) > 9 else 0)") && \
    if [[ "$SHOULD_INSTALL_COLOSSAL" = "0" ]]; then \
        python -c "fname = 'requirements/pytorch/strategies.txt' ; lines = [line for line in open(fname).readlines() if 'colossalai' not in line] ; open(fname, 'w').writelines(lines)" ; \
    fi && \
    echo "$SHOULD_INSTALL_COLOSSAL" && \
    cat requirements/pytorch/strategies.txt && \
    pip install -r requirements/pytorch/devel.txt -r requirements/pytorch/strategies.txt --no-cache-dir --find-links https://download.pytorch.org/whl/cu${CUDA_VERSION_MM}/torch_stable.html

COPY requirements/pytorch/check-avail-extras.py check-avail-extras.py
COPY requirements/pytorch/check-avail-strategies.py check-avail-strategies.py

RUN \
    # Show what we have
    pip --version && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__" && \
    python requirements/pytorch/check-avail-extras.py && \
    python requirements/pytorch/check-avail-strategies.py && \
    rm -rf requirements/
