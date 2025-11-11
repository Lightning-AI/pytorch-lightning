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

import logging

from typing_extensions import override

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.imports import _raise_enterprise_not_available

log = logging.getLogger(__name__)


class KubeflowEnvironment(ClusterEnvironment):
    """Environment for distributed training using the `PyTorchJob`_ operator from `Kubeflow`_.

    This environment, unlike others, does not get auto-detected and needs to be passed to the Fabric/Trainer
    constructor manually.

    .. _PyTorchJob: https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/pytorch/
    .. _Kubeflow: https://www.kubeflow.org

    """

    def __init__(self) -> None:
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.environments.kubeflow import (
            KubeflowEnvironment as EnterpriseKubeflowEnvironment,
        )

        self.kubeflow_impl = EnterpriseKubeflowEnvironment()

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return self.kubeflow_impl.creates_processes_externally

    @property
    @override
    def main_address(self) -> str:
        return self.kubeflow_impl.main_address

    @property
    @override
    def main_port(self) -> int:
        return self.kubeflow_impl.main_port

    @staticmethod
    @override
    def detect() -> bool:
        raise NotImplementedError("The Kubeflow environment can't be detected automatically.")

    @override
    def world_size(self) -> int:
        return self.kubeflow_impl.world_size()

    @override
    def set_world_size(self, size: int) -> None:
        return self.kubeflow_impl.set_world_size(size)

    @override
    def global_rank(self) -> int:
        return self.kubeflow_impl.global_rank()

    @override
    def set_global_rank(self, rank: int) -> None:
        return self.kubeflow_impl.set_global_rank(rank)

    @override
    def local_rank(self) -> int:
        return self.kubeflow_impl.local_rank()

    @override
    def node_rank(self) -> int:
        return self.kubeflow_impl.node_rank()
