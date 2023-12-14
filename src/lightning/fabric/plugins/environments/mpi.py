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
import socket
from functools import lru_cache
from typing import Optional

from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.environments.lightning import find_free_network_port

log = logging.getLogger(__name__)

_MPI4PY_AVAILABLE = RequirementCache("mpi4py")


class MPIEnvironment(ClusterEnvironment):
    """An environment for running on clusters with processes created through MPI.

    Requires the installation of the `mpi4py` package. See also: https://github.com/mpi4py/mpi4py

    """

    def __init__(self) -> None:
        if not _MPI4PY_AVAILABLE:
            raise ModuleNotFoundError(str(_MPI4PY_AVAILABLE))

        from mpi4py import MPI

        self._comm_world = MPI.COMM_WORLD
        self._comm_local: Optional[MPI.Comm] = None
        self._node_rank: Optional[int] = None
        self._main_address: Optional[str] = None
        self._main_port: Optional[int] = None

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return True

    @property
    @override
    def main_address(self) -> str:
        if self._main_address is None:
            self._main_address = self._get_main_address()
        return self._main_address

    @property
    @override
    def main_port(self) -> int:
        if self._main_port is None:
            self._main_port = self._get_main_port()
        return self._main_port

    @staticmethod
    @override
    def detect() -> bool:
        """Returns ``True`` if the `mpi4py` package is installed and MPI returns a world size greater than 1."""
        if not _MPI4PY_AVAILABLE:
            return False

        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size() > 1

    @override
    @lru_cache(1)
    def world_size(self) -> int:
        return self._comm_world.Get_size()

    @override
    def set_world_size(self, size: int) -> None:
        log.debug("MPIEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    @override
    @lru_cache(1)
    def global_rank(self) -> int:
        return self._comm_world.Get_rank()

    @override
    def set_global_rank(self, rank: int) -> None:
        log.debug("MPIEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    @override
    @lru_cache(1)
    def local_rank(self) -> int:
        if self._comm_local is None:
            self._init_comm_local()
        assert self._comm_local is not None
        return self._comm_local.Get_rank()

    @override
    def node_rank(self) -> int:
        if self._node_rank is None:
            self._init_comm_local()
        assert self._node_rank is not None
        return self._node_rank

    def _get_main_address(self) -> str:
        return self._comm_world.bcast(socket.gethostname(), root=0)

    def _get_main_port(self) -> int:
        return self._comm_world.bcast(find_free_network_port(), root=0)

    def _init_comm_local(self) -> None:
        hostname = socket.gethostname()
        all_hostnames = self._comm_world.gather(hostname, root=0)  # returns None on non-root ranks
        # sort all the hostnames, and find unique ones
        unique_hosts = sorted(set(all_hostnames)) if all_hostnames is not None else []
        unique_hosts = self._comm_world.bcast(unique_hosts, root=0)
        # find the index for this host in the list of hosts:
        self._node_rank = unique_hosts.index(hostname)
        self._comm_local = self._comm_world.Split(color=self._node_rank)
