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

FROM nvcr.io/nvidia/pytorch:20.12-py3

MAINTAINER PyTorchLightning <https://github.com/PyTorchLightning>

ARG LIGHTNING_VERSION=""

COPY ./ ./pytorch-lightning/

# install dependencies
RUN \
    # Disable cache
    #conda install "pip>20.1" && \
    #pip config set global.cache-dir false && \
    if [ -z $LIGHTNING_VERSION ] ; then \
        pip install ./pytorch-lightning --no-cache-dir ; \
        rm -rf pytorch-lightning ; \
    else \
        rm -rf pytorch-lightning ; \
        pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/${LIGHTNING_VERSION}.zip --no-cache-dir ; \
    fi

RUN python --version && \
    pip --version && \
    pip list && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)"

# CMD ["/bin/bash"]
