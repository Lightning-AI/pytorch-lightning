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


class BaguaEnvironment(ClusterEnvironment):
    """Environment for distributed training with `Bagua <https://tutorials.baguasys.com/>`_"""

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    @property
    def main_port(self) -> int:
        return int(os.environ.get("MASTER_PORT", -1))

    @property
    def service_port(self) -> int:
        return int(os.environ.get("BAGUA_SERVICE_PORT", -1))

    @staticmethod
    def detect() -> bool:
        return "BAGUA_SERVICE_PORT" in os.environ

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def set_world_size(self, size: int) -> None:
        log.debug("`BaguaEnvironment.set_world_size` was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    def set_global_rank(self, rank: int) -> None:
        log.debug("`BaguaEnvironment.set_global_rank` was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    def node_rank(self) -> int:
        return int(os.environ.get("NODE_RANK", 0))
