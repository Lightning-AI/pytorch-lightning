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

ARG PYTHON_VERSION=3.10
ARG PYTORCH_VERSION=2.0
ARG CUDA_VERSION=11.8.0

FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

ARG LIGHTNING_VERSION=""

COPY ./ /home/pytorch-lightning/

# install dependencies
RUN \
    cd /home && \
    mv pytorch-lightning/_notebooks notebooks && \
    mv pytorch-lightning/examples . && \
    # replace by specific version if asked
    if [ ! -z "$LIGHTNING_VERSION" ] ; then \
        rm -rf pytorch-lightning ; \
        wget https://github.com/Lightning-AI/pytorch-lightning/archive/${LIGHTNING_VERSION}.zip --progress=bar:force:noscroll ; \
        unzip ${LIGHTNING_VERSION}.zip ; \
        mv pytorch-lightning-*/ pytorch-lightning ; \
        rm *.zip ; \
    fi && \
    # otherwise there is collision with folder name and pkg name on Pypi
    cd pytorch-lightning && \
    pip install setuptools==59.5.0 && \
    PACKAGE_NAME=lightning pip install '.[extra,loggers,strategies]' --no-cache-dir && \
    PACKAGE_NAME=pytorch pip install '.[extra,loggers,strategies]' --no-cache-dir && \
    cd .. && \
    rm -rf pytorch-lightning

RUN python --version && \
    pip --version && \
    pip list && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)"

# CMD ["/bin/bash"]
