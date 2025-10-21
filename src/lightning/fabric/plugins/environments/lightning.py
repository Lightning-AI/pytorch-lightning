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

import os
import socket

from filelock import FileLock
from typing_extensions import override

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.rank_zero import rank_zero_only

BASE_PORT = 10000
MAX_PORT = 65000
STEP = 20
LOCK_FILE = "lightning_ports.lock"


class LightningEnvironment(ClusterEnvironment):
    """The default environment used by Lightning for a single node or free cluster (not managed).

    There are two modes the Lightning environment can operate with:

    1.  The user only launches the main process by :code:`python train.py ...` with no additional environment variables
        set. Lightning will spawn new worker processes for distributed training in the current node.
    2.  The user launches all processes manually or with utilities like :code:`torch.distributed.launch`.
        The appropriate environment variables need to be set, and at minimum :code:`LOCAL_RANK`.

    If the main address and port are not provided, the default environment will choose them
    automatically. It is recommended to use this default environment for single-node distributed
    training as it provides a convenient way to launch the training script.

    """

    def __init__(self) -> None:
        super().__init__()
        self._main_port: int = -1
        self._global_rank: int = 0
        self._world_size: int = 1

    @property
    @override
    def creates_processes_externally(self) -> bool:
        """Returns whether the cluster creates the processes or not.

        If at least :code:`LOCAL_RANK` is available as environment variable, Lightning assumes the user acts as the
        process launcher/job scheduler and Lightning will not launch new processes.

        """
        return "LOCAL_RANK" in os.environ

    @property
    @override
    def main_address(self) -> str:
        return os.environ.get("MASTER_ADDR", "127.0.0.1")

    @property
    @override
    def main_port(self) -> int:
        if self._main_port == -1:
            self._main_port = (
                int(os.environ["MASTER_PORT"]) if "MASTER_PORT" in os.environ else find_free_network_port()
            )
        return self._main_port

    @staticmethod
    @override
    def detect() -> bool:
        return True

    @override
    def world_size(self) -> int:
        return self._world_size

    @override
    def set_world_size(self, size: int) -> None:
        self._world_size = size

    @override
    def global_rank(self) -> int:
        return self._global_rank

    @override
    def set_global_rank(self, rank: int) -> None:
        self._global_rank = rank
        rank_zero_only.rank = rank

    @override
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @override
    def node_rank(self) -> int:
        group_rank = os.environ.get("GROUP_RANK", 0)
        return int(os.environ.get("NODE_RANK", group_rank))

    @override
    def teardown(self) -> None:
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.connect_ex(("localhost", port)) != 0


def find_free_network_port(base: int = BASE_PORT, step: int = STEP) -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    MASTER_PORT environment variable.

    """
    PL_FORCE_DETERMINISTIC_PORTS = os.environ.get("PL_FORCE_DETERMINISTIC_PORTS", "0")

    if PL_FORCE_DETERMINISTIC_PORTS == "0":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    # use the last assigned port + step strategy with a file lock to avoid race conditions
    lock_path = os.path.join(os.getcwd(), LOCK_FILE)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    with FileLock(lock_path + ".lock"):
        # read used ports
        if os.path.exists(lock_path):
            with open(lock_path) as f:
                used = [int(x.strip()) for x in f if x.strip()]
        else:
            used = []

        candidate = base if not used else used[-1] + step
        if candidate > MAX_PORT:
            candidate = base

        tries = 0
        max_tries = (MAX_PORT - base) // step
        while (not is_port_available(candidate) or candidate in used) and tries < max_tries:
            candidate += step
            if candidate > MAX_PORT:
                candidate = base
            tries += 1

        if tries >= max_tries:
            raise RuntimeError("No free port found in range")

        # write the new port to the file
        with open(lock_path, "a") as f:
            f.write(f"{candidate}\n")

    return candidate
