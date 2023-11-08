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

FROM ubuntu:20.04

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

SHELL ["/bin/bash", "-c"]
# https://techoverflow.net/2019/05/18/how-to-fix-configuring-tzdata-interactive-input-when-building-docker-images/
ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    PATH="$PATH:/root/.local/bin" \
    # MAKEFLAGS="-j$(nproc)"
    MAKEFLAGS="-j2"

RUN \
    apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        ca-certificates \
        pkg-config \
        cmake \
        git \
        wget \
        curl \
        unzip \
        rsync \
        ninja-build \
        pandoc \
        texlive-latex-extra \
        dvipng \
        texlive-pictures \
        python3 \
        python3-distutils \
        python3-dev \
    && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/docs.txt docs.txt

# NOTE: exclude specific `botocore` below as is seems to be corrupted version
RUN \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install awscli gsutil --no-cache-dir && \
    pip install torch -r docs.txt --no-cache-dir \
      -f https://download.pytorch.org/whl/cpu/torch_stable.html  && \
    pip list
