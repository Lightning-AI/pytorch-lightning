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

FROM google/cloud-sdk:slim

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

# CALL: docker image build -t pytorch-lightning:XLA-image -f dockers/base-xla/Dockerfile . --build-arg PYTHON_VERSION=3.8
ARG PYTHON_VERSION=3.9
ARG CONDA_VERSION=4.9.2
ARG XLA_VERSION=1.12

SHELL ["/bin/bash", "-c"]
# for skipping configurations
ENV \
    DEBIAN_FRONTEND="noninteractive" \
    CONDA_ENV="lightning"

# show system info
RUN lsb_release -a && cat /etc/*-release

RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        curl \
        unzip \
        ca-certificates \
        libomp5 \
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

RUN conda create -y --name $CONDA_ENV && \
    conda init bash && \
    python -c "import re ; fname = 'environment.yml' ; req = re.sub(r'python>=[\d\.]+', 'python=${PYTHON_VERSION}', open(fname).read()) ; open(fname, 'w').write(req)" && \
    # drop unnecessary packages and channels
    python -c "fname = 'environment.yml' ; req = open(fname).readlines() ; open(fname, 'w').writelines([ln for ln in req if not any(n in ln for n in ['pytorch>', 'nvidia', 'torchvision'])])" && \
    cat environment.yml && \
    conda env update --file environment.yml && \
    conda clean -ya && \
    rm environment.yml

ENV \
    PATH="/root/miniconda3/envs/${CONDA_ENV}/bin:$PATH" \
    LD_LIBRARY_PATH="/root/miniconda3/envs/${CONDA_ENV}/lib:$LD_LIBRARY_PATH" \
    # if you want this environment to be the default one, uncomment the following line:
    CONDA_DEFAULT_ENV=${CONDA_ENV}

# Disable cache
RUN pip --version && \
    pip config set global.cache-dir false

# Get package
COPY ./ ./lightning/

RUN \
    python --version && \
    cd lightning && \
    # Pin mkl version to avoid OSError on torch import
    # OSError: libmkl_intel_lp64.so.1: cannot open shared object file: No such file or directory
    # https://github.com/pytorch/xla/issues/1666
    pip install mkl==2021.4.0 && \
    # set particular PyTorch version \
    for fpath in `ls requirements/**/*.txt`; do \
      python requirements/pytorch/adjust-versions.py $fpath ${XLA_VERSION}; \
    done && \
    # install PL dependencies
    pip install --requirement requirements/pytorch/devel.txt && \
    # Install PyTorch XLA
    py_version=${PYTHON_VERSION/./} && \
    pip install "https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-${XLA_VERSION}-cp${py_version}-cp${py_version}-linux_x86_64.whl" && \
    cd .. && \
    rm -rf lightning && \
    rm -rf /root/.cache \

RUN \
    # Show what we have
    conda info && \
    pip list && \
    python -c "import sys; assert sys.version[:3] == '$PYTHON_VERSION', sys.version" && \
    python -c "import torch; assert torch.__version__.startswith('$XLA_VERSION'), torch.__version__"
