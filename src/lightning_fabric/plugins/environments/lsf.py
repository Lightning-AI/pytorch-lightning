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
import socket
from typing import Dict, List

from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.cloud_io import get_filesystem

log = logging.getLogger(__name__)


class LSFEnvironment(ClusterEnvironment):
    """An environment for running on clusters managed by the LSF resource manager.

    It is expected that any execution using this ClusterEnvironment was executed
    using the Job Step Manager i.e. ``jsrun``.

    This plugin expects the following environment variables:

    ``LSB_JOBID``
      The LSF assigned job ID

    ``LSB_DJOB_RANKFILE``
      The OpenMPI compatible rank file for the LSF job

    ``JSM_NAMESPACE_LOCAL_RANK``
      The node local rank for the task. This environment variable is set by ``jsrun``

    ``JSM_NAMESPACE_SIZE``
      The world size for the task. This environment variable is set by ``jsrun``

    ``JSM_NAMESPACE_RANK``
      The global rank for the task. This environment variable is set by ``jsrun``
    """

    def __init__(self) -> None:
        super().__init__()
        self._main_address = self._get_main_address()
        self._main_port = self._get_main_port()
        self._node_rank = self._get_node_rank()
        self._set_init_progress_group_env_vars()

    def _set_init_progress_group_env_vars(self) -> None:
        # set environment variables needed for initializing torch distributed process group
        os.environ["MASTER_ADDR"] = str(self._main_address)
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        os.environ["MASTER_PORT"] = str(self._main_port)
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

    @property
    def creates_processes_externally(self) -> bool:
        """LSF creates subprocesses, i.e., PyTorch Lightning does not need to spawn them."""
        return True

    @property
    def main_address(self) -> str:
        """The main address is read from an OpenMPI host rank file in the environment variable
        ``LSB_DJOB_RANKFILE``."""
        return self._main_address

    @property
    def main_port(self) -> int:
        """The main port is calculated from the LSF job ID."""
        return self._main_port

    @staticmethod
    def detect() -> bool:
        """Returns ``True`` if the current process was launched using the ``jsrun`` command."""
        required_env_vars = {"LSB_JOBID", "LSB_DJOB_RANKFILE", "JSM_NAMESPACE_LOCAL_RANK", "JSM_NAMESPACE_SIZE"}
        return required_env_vars.issubset(os.environ.keys())

    def world_size(self) -> int:
        """The world size is read from the environment variable ``JSM_NAMESPACE_SIZE``."""
        world_size = os.environ.get("JSM_NAMESPACE_SIZE")
        if world_size is None:
            raise ValueError(
                "Cannot determine world size. Environment variable `JSM_NAMESPACE_SIZE` not found."
                "Make sure you run your executable with `jsrun`."
            )
        return int(world_size)

    def set_world_size(self, size: int) -> None:
        log.debug("LSFEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        """The world size is read from the environment variable ``JSM_NAMESPACE_RANK``."""
        global_rank = os.environ.get("JSM_NAMESPACE_RANK")
        if global_rank is None:
            raise ValueError(
                "Cannot determine global rank. Environment variable `JSM_NAMESPACE_RANK` not found."
                "Make sure you run your executable with `jsrun`."
            )
        return int(global_rank)

    def set_global_rank(self, rank: int) -> None:
        log.debug("LSFEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        """The local rank is read from the environment variable `JSM_NAMESPACE_LOCAL_RANK`."""
        local_rank = os.environ.get("JSM_NAMESPACE_LOCAL_RANK")
        if local_rank is None:
            raise ValueError(
                "Cannot determine local rank. Environment variable `JSM_NAMESPACE_LOCAL_RANK` not found."
                "Make sure you run your executable with `jsrun`."
            )
        return int(local_rank)

    def node_rank(self) -> int:
        """The node rank is determined by the position of the current hostname in the OpenMPI host rank file stored
        in ``LSB_DJOB_RANKFILE``."""
        return self._node_rank

    def _get_node_rank(self) -> int:
        """A helper method for getting the node rank.

        The node rank is determined by the position of the current node in the list of hosts used in the job. This is
        calculated by reading all hosts from ``LSB_DJOB_RANKFILE`` and finding this node's hostname in the list.
        """
        hosts = self._read_hosts()
        count: Dict[str, int] = {}
        for host in hosts:
            if host not in count:
                count[host] = len(count)
        return count[socket.gethostname()]

    @staticmethod
    def _read_hosts() -> List[str]:
        """Read compute hosts that are a part of the compute job.

        LSF uses the Job Step Manager (JSM) to manage job steps. Job steps are executed by the JSM from "launch" nodes.
        Each job is assigned a launch node. This launch node will be the first node in the list contained in
        ``LSB_DJOB_RANKFILE``.
        """
        var = "LSB_DJOB_RANKFILE"
        rankfile = os.environ.get(var)
        if rankfile is None:
            raise ValueError("Did not find the environment variable `LSB_DJOB_RANKFILE`")
        if not rankfile:
            raise ValueError("The environment variable `LSB_DJOB_RANKFILE` is empty")

        fs = get_filesystem(rankfile)
        with fs.open(rankfile, "r") as f:
            ret = [line.strip() for line in f]
        # remove the launch node (i.e. the first node in LSB_DJOB_RANKFILE) from the list
        return ret[1:]

    def _get_main_address(self) -> str:
        """A helper for getting the main address.

        The main address is assigned to the first node in the list of nodes used for the job.
        """
        hosts = self._read_hosts()
        return hosts[0]

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
