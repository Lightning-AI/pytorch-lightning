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
from typing import Optional

from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_warn

log = logging.getLogger(__name__)


class TorchElasticEnvironment(ClusterEnvironment):
    """Environment for fault-tolerant and elastic training with `torchelastic <https://pytorch.org/elastic/>`_"""

    @staticmethod
    def is_using_torchelastic() -> bool:
        """Returns ``True`` if the current process was launched using the torchelastic command."""
        required_env_vars = ("RANK", "GROUP_RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        return all(v in os.environ for v in required_env_vars)

    @property
    def creates_processes_externally(self) -> bool:
        return True

    def master_address(self) -> str:
        if "MASTER_ADDR" not in os.environ:
            rank_zero_warn("MASTER_ADDR environment variable is not defined. Set as localhost")
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        master_address = os.environ.get("MASTER_ADDR")
        return master_address

    def master_port(self) -> int:
        if "MASTER_PORT" not in os.environ:
            rank_zero_warn("MASTER_PORT environment variable is not defined. Set as 12910")
            os.environ["MASTER_PORT"] = "12910"
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        port = int(os.environ.get("MASTER_PORT"))
        return port

    def world_size(self) -> Optional[int]:
        world_size = os.environ.get("WORLD_SIZE")
        return int(world_size) if world_size is not None else world_size

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
