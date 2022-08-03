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


FROM rocm/dev-ubuntu-20.04:latest

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=1.12.0
ARG ROCM_WHL_VERSION=5.1.1

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles \
    PATH="$PATH:/root/.local/bin" \
    MKL_THREADING_LAYER=GNU \
    MAKEFLAGS="-j2"

RUN \
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
        ssh

# Install python
RUN \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

#COPY ./requirements/pytorch/ ./requirements/pytorch/
#COPY ./.actions/assistant.py assistant.py

ENV PYTHONPATH=/usr/lib/python${PYTHON_VERSION}/site-packages

RUN \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    PYTHON_VERSION_MM=$(python -c "print(''.join('$PYTHON_VERSION'.split('.')[:2]))") && \
    PYT37M=$(python3 -c "print('m' if $PYTHON_VERSION==3.7 else '')") && \
    pip config set global.cache-dir false && \
    # Example path - https://download.pytorch.org/whl/rocm5.1.1/torch-1.12.0%2Brocm5.1.1-cp38-cp38-linux_x86_64.whl
    pip install --no-cache-dir https://download.pytorch.org/whl/rocm${ROCM_WHL_VERSION}/torch-${PYTORCH_VERSION}%2Brocm${ROCM_WHL_VERSION}-cp${PYTHON_VERSION_MM}-cp${PYTHON_VERSION_MM}${PYT37M}-linux_x86_64.whl

RUN \
    apt-get purge -y cmake && \
    wget -q https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz && \
    tar -zxvf cmake-3.20.2.tar.gz && \
    cd cmake-3.20.2 && \
    ./bootstrap -- -DCMAKE_USE_OPENSSL=OFF && \
    make && \
    make install && \
    cmake  --version

