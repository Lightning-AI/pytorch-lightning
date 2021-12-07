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
from typing import Dict, List

from pytorch_lightning import _logger as log
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_deprecation


class LSFEnvironment(ClusterEnvironment):
    """An environment for running on clusters managed by the LSF resource manager.

    It is expected that any execution using this ClusterEnvironment was executed
    using the Job Step Manager i.e. ``jsrun``.

    This plugin expects the following environment variables.

    LSB_JOBID:
        The LSF assigned job ID

    LSB_HOSTS:
        The hosts used in the job. This string is expected to have the format "batch <rank_0_host> ...."

    JSM_NAMESPACE_LOCAL_RANK:
        The node local rank for the task. This environment variable is set by jsrun

    JSM_NAMESPACE_SIZE:
        The world size for the task. This environment variable is set by jsrun
    """

    def __init__(self) -> None:
        super().__init__()
        # TODO: remove in 1.7
        if hasattr(self, "is_using_lsf") and callable(self.is_using_lsf):
            rank_zero_deprecation(
                f"`{self.__class__.__name__}.is_using_lsf` has been deprecated in v1.6 and will be removed in v1.7."
                " Implement the static method `detect()` instead (do not forget to add the `@staticmethod` decorator)."
            )
        self._main_address = self._get_main_address()
        self._main_port = self._get_main_port()
        log.debug(f"MASTER_ADDR: {self._main_address}")
        log.debug(f"MASTER_PORT: {self._main_port}")

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        """The main address is read from a list of hosts contained in the environment variable `LSB_HOSTS`."""
        return self._main_address

    @property
    def main_port(self) -> int:
        """The main port gets calculated from the LSF job ID."""
        return self._main_port

    @staticmethod
    def detect() -> bool:
        """Returns ``True`` if the current process was launched using the jsrun command."""
        required_env_vars = {"LSB_JOBID", "LSB_HOSTS", "JSM_NAMESPACE_LOCAL_RANK", "JSM_NAMESPACE_SIZE"}
        return required_env_vars.issubset(os.environ.keys())

    def world_size(self) -> int:
        """The world size is read from the environment variable `JSM_NAMESPACE_SIZE`."""
        var = "JSM_NAMESPACE_SIZE"
        world_size = os.environ.get(var)
        if world_size is None:
            raise ValueError(
                f"Cannot determine world size from environment variable {var}."
                " Make sure you run your executable with `jsrun`"
            )
        return int(world_size)

    def set_world_size(self, size: int) -> None:
        log.debug("LSFEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        """The world size is read from the environment variable `JSM_NAMESPACE_RANK`."""
        var = "JSM_NAMESPACE_RANK"
        global_rank = os.environ.get(var)
        if global_rank is None:
            raise ValueError(
                f"Cannot determine global rank from environment variable {var}."
                " Make sure you run your executable with `jsrun`"
            )
        return int(global_rank)

    def set_global_rank(self, rank: int) -> None:
        log.debug("LSFEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        """The local rank is read from the environment variable `JSM_NAMESPACE_LOCAL_RANK`."""
        var = "JSM_NAMESPACE_LOCAL_RANK"
        local_rank = os.environ.get(var)
        if local_rank is None:
            raise ValueError(
                f"Cannot determine local rank from environment variable {var}."
                " Make sure you run your executable with `jsrun`"
            )
        return int(local_rank)

    def node_rank(self) -> int:
        """The node rank is determined by the position of the current hostname in the list of hosts stored in the
        environment variable `LSB_HOSTS`."""
        hosts = self._read_hosts()
        count: Dict[str, int] = {}
        for host in hosts:
            if "batch" in host or "login" in host:
                continue
            if host not in count:
                count[host] = len(count)
        return count[socket.gethostname()]

    @staticmethod
    def _read_hosts() -> List[str]:
        hosts = os.environ.get("LSB_HOSTS")
        if not hosts:
            raise ValueError("Could not find hosts in environment variable LSB_HOSTS")
        hosts = hosts.split()
        if len(hosts) < 2:
            raise ValueError(
                'Cannot parse hosts from LSB_HOSTS environment variable. Expected format: "batch <rank_0_host> ..."'
            )
        return hosts

    def _get_main_address(self) -> str:
        hosts = self._read_hosts()
        return hosts[1]

    @staticmethod
    def _get_main_port() -> int:
        """A helper function for accessing the main port.

        Uses the LSF job ID so all ranks can compute the main port.
        """
        # check for user-specified main port
        if "MASTER_PORT" in os.environ:
            log.debug(f"Using externally specified main port: {os.environ['MASTER_PORT']}")
            return int(os.environ["MASTER_PORT"])
        if "LSB_JOBID" in os.environ:
            port = int(os.environ["LSB_JOBID"])
            # all ports should be in the 10k+ range
            port = port % 1000 + 10000
            log.debug(f"calculated LSF main port: {port}")
            return port
        raise ValueError("Could not find job id in environment variable LSB_JOBID")
