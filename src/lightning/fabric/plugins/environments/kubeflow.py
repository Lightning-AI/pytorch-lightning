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
import os

from typing_extensions import override

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment

log = logging.getLogger(__name__)


class KubeflowEnvironment(ClusterEnvironment):
    """Environment for distributed training using the `PyTorchJob`_ operator from `Kubeflow`_.

    This environment, unlike others, does not get auto-detected and needs to be passed to the Fabric/Trainer
    constructor manually.

    .. _PyTorchJob: https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/pytorch/
    .. _Kubeflow: https://www.kubeflow.org

    """

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return True

    @property
    @override
    def main_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    @property
    @override
    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    @staticmethod
    @override
    def detect() -> bool:
        raise NotImplementedError("The Kubeflow environment can't be detected automatically.")

    @override
    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    @override
    def set_world_size(self, size: int) -> None:
        log.debug("KubeflowEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    @override
    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    @override
    def set_global_rank(self, rank: int) -> None:
        log.debug("KubeflowEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    @override
    def local_rank(self) -> int:
        return 0

    @override
    def node_rank(self) -> int:
        return self.global_rank()
