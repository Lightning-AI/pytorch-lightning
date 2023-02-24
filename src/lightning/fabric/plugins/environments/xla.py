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
from typing import Any

from lightning.fabric.accelerators.tpu import _XLA_AVAILABLE, TPUAccelerator
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment

log = logging.getLogger(__name__)


class XLAEnvironment(ClusterEnvironment):
    """Cluster environment for training on a TPU Pod with the `PyTorch/XLA <https://pytorch.org/xla>`_ library.

    A list of environment variables set by XLA can be found
    `here <https://github.com/pytorch/xla/blob/master/torch_xla/core/xla_env_vars.py>`_.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(*args, **kwargs)

    @property
    def creates_processes_externally(self) -> bool:
        return False

    @property
    def main_address(self) -> str:
        import torch_xla.core.xla_env_vars as xenv

        return os.environ[xenv.TPU_MESH_CTLER_ADDR]

    @property
    def main_port(self) -> int:
        import torch_xla.core.xla_env_vars as xenv

        return int(os.environ[xenv.TPU_MESH_CTLER_PORT])

    @staticmethod
    def detect() -> bool:
        return TPUAccelerator.is_available()

    def world_size(self) -> int:
        import torch_xla.core.xla_model as xm

        return xm.xrt_world_size()

    def set_world_size(self, size: int) -> None:
        log.debug("XLAEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        import torch_xla.core.xla_model as xm

        return xm.get_ordinal()

    def set_global_rank(self, rank: int) -> None:
        log.debug("XLAEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        import torch_xla.core.xla_model as xm

        return xm.get_local_ordinal()

    def node_rank(self) -> int:
        import torch_xla.core.xla_env_vars as xenv

        return int(os.environ.get(xenv.HOST_ORDINAL, 0))
