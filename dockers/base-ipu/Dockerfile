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

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

ARG PYTHON_VERSION=3.9

SHELL ["/bin/bash", "-c"]

# for skipping configurations
ENV \
    DEBIAN_FRONTEND=noninteractive \
    MKL_THREADING_LAYER=GNU

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
        software-properties-common \
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

COPY ./requirements/pytorch/* ./requirements/pytorch/
COPY ./.actions/assistant.py assistant.py

ENV PYTHONPATH=/usr/lib/python${PYTHON_VERSION}/site-packages

RUN \
    # install pip
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    # install assistant.py's dependency
    pip install -q fire && \
    # disable cache \
    pip config set global.cache-dir false && \
    # set particular PyTorch version
    python ./requirements/pytorch/adjust-versions.py requirements/pytorch/base.txt ${PYTORCH_VERSION} && \
    python ./requirements/pytorch/adjust-versions.py requirements/pytorch/extra.txt ${PYTORCH_VERSION} && \
    python ./requirements/pytorch/adjust-versions.py requirements/pytorch/examples.txt ${PYTORCH_VERSION} && \
    # install all requirements
    pip install -r requirements/pytorch/devel.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu && \
    rm assistant.py && \
    rm -rf requirements/ && \
    python -c "import torch; print(torch.__version__)" && \

RUN \
    # Show what we have
    pip --version && \
    conda info && \
    pip list && \
    python -c "import sys; assert sys.version[:3] == '$PYTHON_VERSION', sys.version" && \
    python -c "import torch; assert torch.__version__.startswith('$PYTORCH_VERSION'), torch.__version__"
