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
ARG PYTORCH_VERSION=1.8

FROM pytorchlightning/pytorch_lightning:base-xla-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}

LABEL maintainer="PyTorchLightning <https://github.com/PyTorchLightning>"

COPY ./ ./pytorch-lightning/

# Pull the legacy checkpoints
RUN cd pytorch-lightning && \
    wget https://pl-public-data.s3.amazonaws.com/legacy/checkpoints.zip -P legacy/ && \
    unzip -o legacy/checkpoints.zip -d legacy/ && \
    ls -l legacy/checkpoints/

RUN \
    # drop unnecessary packages
    python .github/prune-packages.py ./pytorch-lightning/requirements/extra.txt "fairscale" "horovod" && \
    pip install -r pytorch-lightning/requirements/devel.txt --no-cache-dir

COPY ./dockers/tpu-tests/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
