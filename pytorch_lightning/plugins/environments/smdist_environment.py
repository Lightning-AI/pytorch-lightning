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

from pytorch_lightning import _logger as log
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.utilities import _SMDIST_AVAILABLE, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _SMDIST_AVAILABLE:
    import smdistributed.dataparallel.torch.distributed as dist


class SMDistributedEnvironment(ClusterEnvironment):

    def __init__(self):
        if not _SMDIST_AVAILABLE:
            raise MisconfigurationException("`smdistributed` module is not available.")
        super().__init__()

    def creates_children(self) -> bool:
        return False

    def master_address(self):
        master_address = os.environ["SM_CURRENT_HOST"]
        log.debug(f"MASTER_ADDR: {master_address}")
        return master_address

    def master_port(self):
        if "MASTER_PORT" not in os.environ:
            rank_zero_warn("MASTER_PORT environment variable is not defined. Set as 12910")
            os.environ["MASTER_PORT"] = "12910"
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        port = os.environ.get("MASTER_PORT")
        return port

    def world_size(self) -> int:
        return dist.get_world_size()

    def local_rank(self) -> int:
        return dist.get_local_rank()

    def node_rank(self) -> int:
        hosts = os.environ["SM_HOSTS"]
        current_host = os.environ["SM_CURRENT_HOST"]
        return hosts.index(current_host) if current_host in hosts else 0

    def global_rank(self) -> int:
        return dist.get_rank()
