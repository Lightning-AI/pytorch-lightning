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

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=1.11
ARG CUDA_VERSION=11.3.1

FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}-cuda${CUDA_VERSION}

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

ARG LIGHTNING_VERSION=""

COPY ./ /home/lightning/

ENV PACKAGE_NAME=pytorch

# install dependencies
RUN \
    cd /home && \
    mv lightning/_notebooks notebooks && \
    mv lightning/examples . && \
    # replace by specific version if asked
    if [ ! -z "$LIGHTNING_VERSION" ] ; then \
        rm -rf lightning ; \
        wget https://github.com/Lightning-AI/lightning/archive/${LIGHTNING_VERSION}.zip --progress=bar:force:noscroll ; \
        unzip ${LIGHTNING_VERSION}.zip ; \
        mv lightning-*/ lightning ; \
        rm *.zip ; \
    fi && \
    # otherwise there is collision with folder name ans pkg name on Pypi
    cd lightning && \
    SHOULD_INSTALL_COLOSSAL=$(python -c "import torch; print(1 if int(torch.__version__.split('.')[1]) > 9 else 0)") && \
    if [[ "$SHOULD_INSTALL_COLOSSAL" = "0" ]]; then \
        python -c "fname = 'requirements/pytorch/strategies.txt' ; lines = [line for line in open(fname).readlines() if 'colossalai' not in line] ; open(fname, 'w').writelines(lines)" ; \
    fi && \
    pip install .["extra","loggers","strategies"] --no-cache-dir --find-links https://release.colossalai.org && \
    cd .. && \
    rm -rf lightning

RUN python --version && \
    pip --version && \
    pip list && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)"

# CMD ["/bin/bash"]
