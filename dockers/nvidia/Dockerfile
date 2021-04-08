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

FROM nvcr.io/nvidia/cuda:11.1.1-runtime-ubuntu20.04

MAINTAINER PyTorchLightning <https://github.com/PyTorchLightning>

ARG LIGHTNING_VERSION=""

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Prague \
    PATH="$PATH:/root/.local/bin" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    MKL_THREADING_LAYER=GNU

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3 \
        python3-distutils \
        python3-dev \
        pkg-config \
        cmake \
        git \
        wget \
        unzip \
        ca-certificates \
    && \

# Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/* && \

# Setup PIP
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python get-pip.py && \
    rm get-pip.py && \
    pip --version

COPY ./ /home/pytorch-lightning/

RUN \
    cd /home  && \
    mv pytorch-lightning/notebooks . && \
    mv pytorch-lightning/pl_examples . && \
    # replace by specific version if asked
    if [ ! -z "$LIGHTNING_VERSION" ] ; then \
        rm -rf pytorch-lightning ; \
        wget https://github.com/PyTorchLightning/pytorch-lightning/archive/${LIGHTNING_VERSION}.zip --progress=bar:force:noscroll ; \
        unzip ${LIGHTNING_VERSION}.zip ; \
        mv pytorch-lightning-*/ pytorch-lightning ; \
        rm *.zip ; \
    fi && \

# Installations
    python -c "fname = './pytorch-lightning/requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('horovod')] ; open(fname, 'w').writelines(lines)" && \
    pip install -r ./pytorch-lightning/requirements/extra.txt -U --no-cache-dir && \
    pip install -r ./pytorch-lightning/requirements/examples.txt -U --no-cache-dir && \
    pip install ./pytorch-lightning --no-cache-dir && \
    rm -rf pytorch-lightning

RUN python --version && \
    pip --version && \
    pip list && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)"

# CMD ["/bin/bash"]
