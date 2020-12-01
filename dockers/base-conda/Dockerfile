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

# Existing images:
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.8 --build-arg PYTORCH_CHANNEL=pytorch-nightly
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.6 --build-arg PYTORCH_CHANNEL=pytorch
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.5 --build-arg PYTORCH_CHANNEL=pytorch
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.4 --build-arg PYTORCH_CHANNEL=pytorch
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.3 --build-arg PYTORCH_CHANNEL=pytorch

ARG CUDNN_VERSION=8
ARG CUDA_VERSION=10.2

# FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04
# FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu18.04

ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.6
ARG PYTORCH_CHANNEL=pytorch
ARG CONDA_VERSION=4.7.12

SHELL ["/bin/bash", "-c"]

ENV PATH="$PATH:/root/.local/bin"

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
    && \

# Install conda and python.
# NOTE new Conda does not forward the exit status... https://github.com/conda/conda/issues/8385
    curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh && \

# Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:$PATH"
ENV LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH"
ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"

ENV HOROVOD_GPU_OPERATIONS=NCCL
ENV HOROVOD_WITH_PYTORCH=1
ENV HOROVOD_WITHOUT_TENSORFLOW=1
ENV HOROVOD_WITHOUT_MXNET=1
ENV HOROVOD_WITH_GLOO=1
ENV HOROVOD_WITHOUT_MPI=1
#ENV MAKEFLAGS="-j$(nproc)"
ENV MAKEFLAGS="-j1"
ENV TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;7.5"

ENV CONDA_ENV=lightning
COPY environment.yml environment.yml

# conda init
RUN conda create -y --name $CONDA_ENV cudatoolkit=${CUDA_VERSION} && \
    conda init bash && \
    # NOTE: this requires that the channel is presented in the yaml before packages
    # replace channel to nigtly if needed, fix PT version and remove Horovod as it will be installe later
    python -c "fname = 'environment.yml' ; req = open(fname).read().replace('pytorch', '${PYTORCH_CHANNEL}', 1) ; open(fname, 'w').write(req)" && \
    python -c "import re ; fname = 'environment.yml' ; req = re.sub(r'python[>=]+[\d\.]+', 'python=${PYTHON_VERSION}', open(fname).read()) ; open(fname, 'w').write(req)" && \
    python -c "import re ; fname = 'environment.yml' ; req = re.sub(r'torch[>=]+[\d\.]+', 'torch=${PYTORCH_VERSION}', open(fname).read()) ; open(fname, 'w').write(req)" && \
    python -c "fname = 'environment.yml' ; req = open(fname).readlines() ; open(fname, 'w').writelines([ln for ln in req if 'horovod' not in ln])" && \
    cat environment.yml && \
    conda env update --file environment.yml && \
    conda clean -ya && \
    rm environment.yml

ENV PATH /root/miniconda3/envs/${CONDA_ENV}/bin:$PATH
ENV LD_LIBRARY_PATH="/root/miniconda3/envs/${CONDA_ENV}/lib:$LD_LIBRARY_PATH"
# if you want this environment to be the default one, uncomment the following line:
ENV CONDA_DEFAULT_ENV=${CONDA_ENV}

COPY ./requirements/extra.txt requirements-extra.txt
COPY ./requirements/test.txt requirements-test.txt

RUN \
    # Disable cache
    pip config set global.cache-dir false && \
    # Drop fairscale for PT <= 1.4
    if [[ $PYTORCH_VERSION < 1.4 ]] ; then \
        python -c "fname = 'requirements-extra.txt' ; lines = [line for line in open(fname).readlines() if 'fairscale' not in line] ; open(fname, 'w').writelines(lines)" ; \
    fi && \
    # Install remaining requirements
    pip install -r requirements-extra.txt --upgrade-strategy only-if-needed && \
    pip install -r requirements-test.txt --upgrade-strategy only-if-needed && \
    rm requirements*

RUN \
    # install DALI, needed for examples
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda${CUDA_VERSION%%.*}0

RUN \
    # install NVIDIA AMP
    git clone https://github.com/NVIDIA/apex && \
    pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex && \
    rm -rf apex

RUN \
    # Show what we have
    pip --version && \
    conda info && \
    pip list && \
    python -c "import sys; assert sys.version[:3] == '$PYTHON_VERSION', sys.version" && \
    python -c "import torch; assert torch.__version__[:3] == '$PYTORCH_VERSION', torch.__version__"
