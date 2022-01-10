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

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu18.04

ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.6
ARG CONDA_VERSION=4.9.2

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    PATH="$PATH:/root/.local/bin" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Prague \
    # CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER=GNU

RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        libopenmpi-dev \
    && \

# Install conda and python.
# NOTE new Conda does not forward the exit status... https://github.com/conda/conda/issues/8385
    curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_${CONDA_VERSION}-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh && \

# Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

ENV \
    PATH="/root/miniconda3/bin:$PATH" \
    LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER=GNU \
    MAKEFLAGS="-j$(nproc)" \
    # MAKEFLAGS="-j1" \
    TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5" \
    CONDA_ENV=lightning

COPY environment.yml environment.yml

# conda init
RUN conda update -n base -c defaults conda && \
    conda create -y --name $CONDA_ENV python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION} cudatoolkit=${CUDA_VERSION} -c nvidia -c pytorch -c pytorch-test -c pytorch-nightly && \
    conda init bash && \
    # NOTE: this requires that the channel is presented in the yaml before packages
    python -c "import re ; fname = 'environment.yml' ; req = re.sub(r'- python[>=]+[\d\.]+', '# - python=${PYTHON_VERSION}', open(fname).read()) ; open(fname, 'w').write(req)" && \
    python -c "import re ; fname = 'environment.yml' ; req = re.sub(r'- pytorch[>=]+[\d\.]+', '# - pytorch=${PYTORCH_VERSION}', open(fname).read()) ; open(fname, 'w').write(req)" && \
    cat environment.yml && \
    conda env update --name $CONDA_ENV --file environment.yml && \
    conda clean -ya && \
    rm environment.yml

ENV \
    PATH=/root/miniconda3/envs/${CONDA_ENV}/bin:$PATH \
    LD_LIBRARY_PATH="/root/miniconda3/envs/${CONDA_ENV}/lib:$LD_LIBRARY_PATH" \
    # if you want this environment to be the default o \ne, uncomment the following line:
    CONDA_DEFAULT_ENV=${CONDA_ENV} \
    HOROVOD_CUDA_HOME=$CUDA_TOOLKIT_ROOT_DIR \
    HOROVOD_BUILD_CUDA_CC_LIST=$TORCH_CUDA_ARCH_LIST \
    HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    HOROVOD_WITH_GLOO=1 \
    HOROVOD_WITHOUT_MPI=1

COPY ./requirements/extra.txt requirements-extra.txt
COPY ./requirements/examples.txt requirements-examples.txt
COPY ./requirements/test.txt requirements-test.txt
COPY ./requirements/adjust_versions.py requirements_adjust_versions.py

RUN \
    pip list | grep torch && \
    python -c "import torch; print(torch.__version__)" && \
    python requirements_adjust_versions.py requirements-extra.txt && \
    python requirements_adjust_versions.py requirements-examples.txt && \
    # Install remaining requirements
    pip install -r requirements-extra.txt --no-cache-dir && \
    pip install -r requirements-examples.txt --no-cache-dir --find-links https://download.pytorch.org/whl/nightly/torch_nightly.html && \
    pip install -r requirements-test.txt --no-cache-dir && \
    rm requirements*

RUN \
    CUDA_VERSION_MAJOR=$(python -c "import torch ; print(torch.version.cuda.split('.')[0])") && \
    py_ver=$(python -c "print(int('$PYTHON_VERSION'.split('.') >= '3.9'.split('.')))") && \
    # install DALI, needed for examples
    # todo: waiting for 1.4 - https://github.com/NVIDIA/DALI/issues/3144#issuecomment-877386691
    if [ $py_ver -eq "0" ]; then \
        pip install --extra-index-url https://developer.download.nvidia.com/compute/redist "nvidia-dali-cuda${CUDA_VERSION_MAJOR}0>1.0" ; \
        python -c 'from nvidia.dali.pipeline import Pipeline' ; \
    fi

RUN \
    # install NVIDIA apex
    pip install --no-cache-dir --global-option="--cuda_ext" https://github.com/NVIDIA/apex/archive/refs/heads/master.zip && \
    python -c "from apex import amp"

RUN \
    # Show what we have
    pip --version && \
    conda info && \
    pip list && \
    python -c "import sys; ver = sys.version_info ; assert f'{ver.major}.{ver.minor}' == '$PYTHON_VERSION', ver" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__"
