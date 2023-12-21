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

ARG PYTORCH_VERSION=22.09

# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

LABEL maintainer="Lightning-AI <https://github.com/Lightning-AI>"

ARG LIGHTNING_VERSION=""

RUN python -c "import torch ; print(torch.__version__)" >> torch_version.info

COPY ./ /workspace/pytorch-lightning/

RUN \
    cd /workspace  && \
    # replace by specific version if asked
    if [ ! -z "$LIGHTNING_VERSION" ] ; then \
        rm -rf pytorch-lightning ; \
        git clone https://github.com/Lightning-AI/pytorch-lightning.git ; \
        cd pytorch-lightning ; \
        git checkout ${LIGHTNING_VERSION} ; \
        git submodule update --init --recursive ; \
        cd .. ; \
    fi && \
# save the examples \
    ls -lh pytorch-lightning/ && \
    rm -rf pytorch-lightning/.git && \
    rm -rf pytorch-lightning/_notebooks/.git && \
    mv pytorch-lightning/_notebooks/.notebooks/ notebooks && \
    cp -r pytorch-lightning/*examples . && \

# Installations \
    pip install "Pillow>=8.2, !=8.3.0" "cryptography>=3.4" "py>=1.10" --no-cache-dir && \
    PACKAGE_NAME=pytorch pip install './lightning[extra,loggers,strategies]' --no-cache-dir && \
    rm -rf pytorch-lightning && \
    pip list

RUN pip install jupyterlab[all] -U --no-cache-dir

# create jupyter_notebook_config.py
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.contents_manager_class = 'notebook.services.contents.largefilemanager.LargeFileManager'" > /root/.jupyter/jupyter_notebook_config.py

ENV PYTHONPATH="/workspace"

RUN \
    TORCH_VERSION=$(cat torch_version.info) && \
    rm torch_version.info && \
    ls -lh . && \
    python --version && \
    pip --version && \
    pip list | grep torch && \
    python -c "import torch; assert torch.__version__.startswith('$TORCH_VERSION'), torch.__version__" && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)"

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
