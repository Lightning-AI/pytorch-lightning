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

FROM ubuntu:20.04

LABEL maintainer="PyTorchLightning <https://github.com/PyTorchLightning>"

ARG PYTHON_VERSION=3.9
ARG CONDA_VERSION=4.9.2

SHELL ["/bin/bash", "-c"]

# for skipping configurations
ENV \
    DEBIAN_FRONTEND=noninteractive \
    CONDA_ENV=lightning

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        jq \
        libopenmpi-dev \
        unzip \
        wget \
    && \
# Install conda and python.
# NOTE new Conda does not forward the exit status... https://github.com/conda/conda/issues/8385
    curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_${CONDA_VERSION}-Linux-x86_64.sh && \
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
    LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH"

COPY environment.yml environment.yml

RUN conda create -y --name $CONDA_ENV python=${PYTHON_VERSION} pytorch=${PYTORCH_VERSION} cudatoolkit=${CUDA_VERSION} -c pytorch && \
    conda init bash && \
    python -c "import re ; fname = 'environment.yml' ; req = re.sub(r'python>=[\d\.]+', 'python=${PYTHON_VERSION}', open(fname).read()) ; open(fname, 'w').write(req)" && \
    python -c "import re ; fname = 'environment.yml' ; req = re.sub(r'- pytorch[>=]+[\d\.]+', '# - pytorch=${PYTORCH_VERSION}', open(fname).read()) ; open(fname, 'w').write(req)" && \
    python -c "fname = 'environment.yml' ; req = open(fname).readlines() ; open(fname, 'w').writelines([ln for ln in req if not any(n in ln for n in ['pytorch>', 'horovod'])])" && \
    cat environment.yml && \
    conda env update --file environment.yml && \
    conda clean -ya && \
    rm environment.yml

ENV \
    PATH=/root/miniconda3/envs/${CONDA_ENV}/bin:$PATH \
    LD_LIBRARY_PATH="/root/miniconda3/envs/${CONDA_ENV}/lib:$LD_LIBRARY_PATH" \
    # if you want this environment to be the default one, uncomment the following line:
    CONDA_DEFAULT_ENV=${CONDA_ENV} \
    MKL_THREADING_LAYER=GNU

COPY ./requirements/extra.txt requirements-extra.txt
COPY ./requirements/test.txt requirements-test.txt
COPY ./requirements/adjust_versions.py adjust_versions.py
COPY ./.github/prune-packages.py prune_packages.py

RUN \
    pip list | grep torch && \
    python -c "import torch; print(torch.__version__)" && \
    python adjust_versions.py requirements-extra.txt && \
    python prune_packages.py requirements-extra.txt "fairscale" "horovod" && \
    # Install remaining requirements
    pip install -r requirements-extra.txt --no-cache-dir && \
    pip install -r requirements-test.txt --no-cache-dir && \
    rm requirements*

RUN \
    # Show what we have
    pip --version && \
    conda info && \
    pip list && \
    python -c "import sys; assert sys.version[:3] == '$PYTHON_VERSION', sys.version" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__"
