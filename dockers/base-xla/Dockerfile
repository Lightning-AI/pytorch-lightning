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

FROM google/cloud-sdk:slim

LABEL maintainer="PyTorchLightning <https://github.com/PyTorchLightning>"

# CALL: docker image build -t pytorch-lightning:XLA-extras-py3.6 -f dockers/base-xla/Dockerfile . --build-arg PYTHON_VERSION=3.6
# This Dockerfile installs pytorch/xla 3.7 wheels. There are also 3.6 wheels available; see below.
ARG PYTHON_VERSION=3.7
ARG XLA_VERSION=1.6

SHELL ["/bin/bash", "-c"]

ARG CONDA_VERSION=4.9.2
# for skipping configurations
ENV \
    DEBIAN_FRONTEND=noninteractive \
    CONDA_ENV=lightning

# show system info
RUN lsb_release -a && cat /etc/*-release

RUN apt-get update -qq && \
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
    LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH"
COPY environment.yml environment.yml

RUN conda create -y --name $CONDA_ENV && \
    conda init bash && \
    # replace channel to nigtly if neede, fix PT version and remove Horovod as it will be installe later
    python -c "import re ; fname = 'environment.yml' ; req = re.sub(r'python>=[\d\.]+', 'python=${PYTHON_VERSION}', open(fname).read()) ; open(fname, 'w').write(req)" && \
    python -c "fname = 'environment.yml' ; req = open(fname).readlines() ; open(fname, 'w').writelines([ln for ln in req if not any(n in ln for n in ['pytorch>', 'horovod'])])" && \
    cat environment.yml && \
    conda env update --file environment.yml && \
    conda clean -ya && \
    rm environment.yml

ENV \
    PATH=/root/miniconda3/envs/${CONDA_ENV}/bin:$PATH \
    LD_LIBRARY_PATH="/root/miniconda3/envs/${CONDA_ENV}/lib:$LD_LIBRARY_PATH" \
    # if you want this environment to be the default one, uncomment the following line:
    CONDA_DEFAULT_ENV=${CONDA_ENV}

# Disable cache
RUN pip --version && \
    pip config set global.cache-dir false && \
    conda remove pytorch torchvision && \
    # Install Pytorch XLA
    py_version=${PYTHON_VERSION/./} && \
    # Python 3.7 wheels are available. Replace cp36-cp36m with cp37-cp37m
    gsutil cp "gs://tpu-pytorch/wheels/torch-${XLA_VERSION}-cp${py_version}-cp${py_version}m-linux_x86_64.whl" . && \
    gsutil cp "gs://tpu-pytorch/wheels/torch_xla-${XLA_VERSION}-cp${py_version}-cp${py_version}m-linux_x86_64.whl" . && \
    gsutil cp "gs://tpu-pytorch/wheels/torchvision-${XLA_VERSION}-cp${py_version}-cp${py_version}m-linux_x86_64.whl" . && \
    pip install *.whl && \
    rm *.whl

# Get package
COPY ./ ./pytorch-lightning/

RUN \
    python --version && \
    cd pytorch-lightning && \
    # drop packages installed with XLA
    python -c "fname = 'requirements.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('torch')] ; open(fname, 'w').writelines(lines)" && \
    python -c "fname = 'requirements/examples.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('torchvision')] ; open(fname, 'w').writelines(lines)" && \
    # drop unnecessary packages
    python -c "fname = 'requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('horovod')] ; open(fname, 'w').writelines(lines)" && \
    python -c "fname = 'requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if 'fairscale' not in line] ; open(fname, 'w').writelines(lines)" && \
    python ./requirements/adjust_versions.py ./requirements/extra.txt && \
    # install PL dependencies
    pip install --requirement ./requirements/devel.txt --no-cache-dir && \
    cd .. && \
    rm -rf pytorch-lightning && \
    rm -rf /root/.cache

RUN \
    # Show what we have
    pip --version && \
    conda info && \
    pip list && \
    python -c "import sys; assert sys.version[:3] == '$PYTHON_VERSION', sys.version" && \
    python -c "import torch; ver = '$XLA_VERSION' ; ver = dict(nightly='1.9').get(ver, ver) ; assert torch.__version__[:3] == ver, torch.__version__"
