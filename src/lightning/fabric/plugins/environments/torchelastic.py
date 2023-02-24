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

import torch.distributed

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.rank_zero import rank_zero_warn

log = logging.getLogger(__name__)


class TorchElasticEnvironment(ClusterEnvironment):
    """Environment for fault-tolerant and elastic training with `torchelastic <https://pytorch.org/elastic/>`_"""

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        if "MASTER_ADDR" not in os.environ:
            rank_zero_warn("MASTER_ADDR environment variable is not defined. Set as localhost")
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        return os.environ["MASTER_ADDR"]

    @property
    def main_port(self) -> int:
        if "MASTER_PORT" not in os.environ:
            rank_zero_warn("MASTER_PORT environment variable is not defined. Set as 12910")
            os.environ["MASTER_PORT"] = "12910"
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        return int(os.environ["MASTER_PORT"])

    @staticmethod
    def detect() -> bool:
        """Returns ``True`` if the current process was launched using the torchelastic command."""
        # if not available (for example on MacOS), `is_torchelastic_launched` is not defined
        return torch.distributed.is_available() and torch.distributed.is_torchelastic_launched()

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def set_world_size(self, size: int) -> None:
        log.debug("TorchElasticEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    def set_global_rank(self, rank: int) -> None:
        log.debug(
            "TorchElasticEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored."
        )

    def local_rank(self) -> int:
        return int(os.environ["LOCAL_RANK"])

    def node_rank(self) -> int:
        return int(os.environ.get("GROUP_RANK", 0))
