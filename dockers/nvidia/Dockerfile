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

ARG PYTORCH_VERSION=21.11

# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

LABEL maintainer="PyTorchLightning <https://github.com/PyTorchLightning>"

ARG LIGHTNING_VERSION=""

RUN python -c "import torch ; print(torch.__version__)" >> torch_version.info

COPY ./ /workspace/pytorch-lightning/

RUN \
    cd /workspace  && \
    # replace by specific version if asked
    if [ ! -z "$LIGHTNING_VERSION" ] ; then \
        rm -rf pytorch-lightning ; \
        git clone https://github.com/PyTorchLightning/pytorch-lightning.git ; \
        cd pytorch-lightning ; \
        git checkout ${LIGHTNING_VERSION} ; \
        git submodule update --init --recursive ; \
        cd .. ; \
    fi && \
# save the examples
    mv pytorch-lightning/_notebooks notebooks && \
    mv pytorch-lightning/pl_examples . && \

# Installations
    python ./pytorch-lightning/.github/prune-packages.py ./pytorch-lightning/requirements/extra.txt "horovod" && \
    pip install "Pillow>=8.2, !=8.3.0" "cryptography>=3.4" "py>=1.10" --no-cache-dir --upgrade-strategy only-if-needed && \
    pip install -r ./pytorch-lightning/requirements/extra.txt --no-cache-dir --upgrade-strategy only-if-needed && \
    pip install -r ./pytorch-lightning/requirements/examples.txt --no-cache-dir --upgrade-strategy only-if-needed && \
    pip install ./pytorch-lightning --no-cache-dir && \
    rm -rf pytorch-lightning && \
    pip install jupyterlab[all] -U && \
    pip list

RUN pip install lightning-grid -U && \
    pip install "py>=1.10" "protobuf>=3.15.6" --upgrade-strategy only-if-needed

ENV PYTHONPATH="/workspace"

RUN \
    TORCH_VERSION=$(cat torch_version.info) && \
    rm torch_version.info && \
    python --version && \
    pip --version && \
    pip list | grep torch && \
    python -c "import torch; assert torch.__version__.startswith('$TORCH_VERSION'), torch.__version__" && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)"

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
