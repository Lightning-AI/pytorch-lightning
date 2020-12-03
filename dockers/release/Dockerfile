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

ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.5

FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}

MAINTAINER PyTorchLightning <https://github.com/PyTorchLightning>

ARG LIGHTNING_VERSION=""

COPY ./ ./pytorch-lightning/

# install dependencies
RUN \
    # Disable cache
    #conda install "pip>20.1" && \
    #pip config set global.cache-dir false && \
    if [ -z $LIGHTNING_VERSION ] ; then \
        pip install ./pytorch-lightning --upgrade-strategy only-if-needed ; \
        rm -rf pytorch-lightning ; \
    else \
        rm -rf pytorch-lightning ; \
        pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/${LIGHTNING_VERSION}.zip --upgrade-strategy only-if-needed ; \
    fi

RUN python --version && \
    pip --version && \
    pip list && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)"

# CMD ["/bin/bash"]
