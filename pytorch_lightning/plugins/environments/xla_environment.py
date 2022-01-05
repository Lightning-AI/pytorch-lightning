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
import os

from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.utilities import _TPU_AVAILABLE

if _TPU_AVAILABLE:
    import torch_xla.core.xla_env_vars as xenv


class XLAEnvironment(ClusterEnvironment):
    """XLAEnvironment

    A list of environment variables set by XLA can be found
    `here <https://github.com/pytorch/xla/blob/master/torch_xla/core/xla_env_vars.py>`_.
    """

    @property
    def creates_processes_externally(self) -> bool:
        return False

    @property
    def main_address(self) -> str:
        return os.environ[xenv.TPU_MESH_CTLER_ADDR]

    @property
    def main_port(self) -> int:
        return int(os.environ[xenv.TPU_MESH_CTLER_PORT])

    @staticmethod
    def detect() -> bool:
        return _TPU_AVAILABLE

    def world_size(self) -> int:
        return int(os.environ.get(xenv.WORLD_SIZE, 1))

    def set_world_size(self, size: int) -> None:
        pass

    def global_rank(self) -> int:
        return int(os.environ.get(xenv.ORDINAL, 0))

    def set_global_rank(self, rank: int) -> None:
        pass

    def local_rank(self) -> int:
        return int(os.environ.get(xenv.LOCAL_ORDINAL, 0))

    def node_rank(self) -> int:
        return int(os.environ.get(xenv.HOST_ORDINAL, 0))
