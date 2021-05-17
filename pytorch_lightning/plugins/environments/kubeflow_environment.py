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

import logging
import os

from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment

log = logging.getLogger(__name__)


class KubeflowEnvironment(ClusterEnvironment):
    """
    Environment for distributed training using the
    `PyTorchJob <https://www.kubeflow.org/docs/components/training/pytorch/>`_
    operator from `Kubeflow <https://www.kubeflow.org>`_
    """

    @staticmethod
    def is_using_kubeflow() -> bool:
        """ Returns ``True`` if the current process was launched using Kubeflow PyTorchJob. """
        required_env_vars = ("KUBERNETES_PORT", "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK")
        # torchelastic sets these. Make sure we're not in torchelastic
        excluded_env_vars = ("GROUP_RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        return (all(v in os.environ for v in required_env_vars) and not any(v in os.environ for v in excluded_env_vars))

    def creates_children(self) -> bool:
        return True

    def master_address(self) -> str:
        return os.environ['MASTER_ADDR']

    def master_port(self) -> int:
        return int(os.environ['MASTER_PORT'])

    def world_size(self) -> int:
        return int(os.environ['WORLD_SIZE'])

    def set_world_size(self, size: int) -> None:
        log.debug("KubeflowEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    def set_global_rank(self, rank: int) -> None:
        log.debug("KubeflowEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        return 0

    def node_rank(self) -> int:
        return self.global_rank()
