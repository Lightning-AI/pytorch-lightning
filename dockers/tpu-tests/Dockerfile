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
ARG PYTORCH_VERSION=1.6

FROM pytorchlightning/pytorch_lightning:base-xla-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}

MAINTAINER PyTorchLightning <https://github.com/PyTorchLightning>

#SHELL ["/bin/bash", "-c"]

COPY ./ ./pytorch-lightning/

# If using this image for tests, intall more dependencies and don"t delete the source code where the tests live.
RUN \
    # Install pytorch-lightning at the current PR, plus dependencies.
    #pip install -r pytorch-lightning/requirements.txt --no-cache-dir && \
    # drop Horovod
    #python -c "fname = 'pytorch-lightning/requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('horovod')] ; open(fname, 'w').writelines(lines)" && \
    pip install -r pytorch-lightning/requirements/devel.txt --no-cache-dir --upgrade-strategy only-if-needed

#RUN python -c "import pytorch_lightning as pl; print(pl.__version__)"

COPY ./dockers/tpu-tests/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
