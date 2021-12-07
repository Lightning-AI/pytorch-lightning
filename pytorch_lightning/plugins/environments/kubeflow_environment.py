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
from pytorch_lightning.utilities import rank_zero_deprecation

log = logging.getLogger(__name__)


class KubeflowEnvironment(ClusterEnvironment):
    """Environment for distributed training using the `PyTorchJob`_ operator from `Kubeflow`_

    .. _PyTorchJob: https://www.kubeflow.org/docs/components/training/pytorch/
    .. _Kubeflow: https://www.kubeflow.org
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: remove in 1.7
        if hasattr(self, "is_using_kubeflow") and callable(self.is_using_kubeflow):
            rank_zero_deprecation(
                f"`{self.__class__.__name__}.is_using_kubeflow` has been deprecated in v1.6 and will be removed in"
                f" v1.7. Implement the static method `detect()` instead (do not forget to add the `@staticmethod`"
                f" decorator)."
            )

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    @property
    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

    @staticmethod
    def detect() -> bool:
        """Returns ``True`` if the current process was launched using Kubeflow PyTorchJob."""
        required_env_vars = {"KUBERNETES_PORT", "MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"}
        # torchelastic sets these. Make sure we're not in torchelastic
        excluded_env_vars = {"GROUP_RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"}
        env_vars = os.environ.keys()
        return required_env_vars.issubset(env_vars) and excluded_env_vars.isdisjoint(env_vars)

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

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
