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

ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.7

FROM pytorchlightning/pytorch_lightning:base-ipu-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}

LABEL maintainer="PyTorchLightning <https://github.com/PyTorchLightning>"

RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

WORKDIR /azp

COPY ./dockers/ipu-ci-runner/start.sh /usr/local/bin/

RUN curl -o /usr/local/bin/installdependencies.sh \
    "https://raw.githubusercontent.com/microsoft/azure-pipelines-agent/d2acd5f77c6b3914cdb6ed0e5fbea672929c7da9/src/Misc/layoutbin/installdependencies.sh" && \
    chmod +x /usr/local/bin/installdependencies.sh && \
    chmod +x /usr/local/bin/start.sh && \
    /usr/local/bin/installdependencies.sh

ENTRYPOINT ["/usr/local/bin/start.sh"]
CMD ["bash"]
