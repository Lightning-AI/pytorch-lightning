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
import socket

from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_only


class LightningEnvironment(ClusterEnvironment):
    """The default environment used by Lightning for a single node or free cluster (not managed).

    There are two modes the Lightning environment can operate with:

    1.  The user only launches the main process by :code:`python train.py ...` with no additional environment variables
        set. Lightning will spawn new worker processes for distributed training in the current node.
    2.  The user launches all processes manually or with utilities like :code:`torch.distributed.launch`.
        The appropriate environment variables need to be set, and at minimum :code:`LOCAL_RANK`.

    If the master address and port are not provided, the default environment will choose them
    automatically. It is recommended to use this default environment for single-node distributed
    training as it provides a convenient way to launch the training script.
    """

    def __init__(self):
        super().__init__()
        self._master_port = None
        self._global_rank: int = 0
        self._world_size: int = 1

    @property
    def creates_processes_externally(self) -> bool:
        """Returns whether the cluster creates the processes or not.

        If at least :code:`LOCAL_RANK` is available as environment variable, Lightning assumes the user acts as the
        process launcher/job scheduler and Lightning will not launch new processes.
        """
        return "LOCAL_RANK" in os.environ

    def master_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    def master_port(self) -> int:
        if self._master_port is None:
            self._master_port = os.environ.get("MASTER_PORT", find_free_network_port())
        return int(self._master_port)

    def world_size(self) -> int:
        return self._world_size

    def set_world_size(self, size: int) -> None:
        self._world_size = size

    def global_rank(self) -> int:
        return self._global_rank

    def set_global_rank(self, rank: int) -> None:
        self._global_rank = rank
        rank_zero_only.rank = rank

    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    def node_rank(self) -> int:
        group_rank = os.environ.get("GROUP_RANK", 0)
        return int(os.environ.get("NODE_RANK", group_rank))

    def teardown(self) -> None:
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real master node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port
