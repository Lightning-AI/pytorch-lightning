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

from pytorch_lightning import _logger as log
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_deprecation


class LSFEnvironment(ClusterEnvironment):
    """An environment for running on clusters managed by the LSF resource manager.

    It is expected that any execution using this ClusterEnvironment was executed
    using the Job Step Manager i.e. ``jsrun``.

    This plugin expects the following environment variables:

    ``LSB_JOBID``
      The LSF assigned job ID

    ``LSB_DJOB_RANKFILE``
      The OpenMPI compatibile rank file for the LSF job

    ``JSM_NAMESPACE_LOCAL_RANK``
      The node local rank for the task. This environment variable is set by ``jsrun``

    ``JSM_NAMESPACE_SIZE``
      The world size for the task. This environment variable is set by ``jsrun``

    ``JSM_NAMESPACE_RANK``
      The global rank for the task. This environment variable is set by ``jsrun``
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
        self._local_rank = self._get_local_rank()
        self._global_rank = self._get_global_rank()
        self._world_size = self._get_world_size()
        self._node_rank = self._get_node_rank()
        self._rep = (
            f"main_address={self._main_address},main_port={self._main_port},local_rank={self._local_rank},"
            f"global_rank={self._global_rank},world_size={self._world_size},node_rank={self._node_rank}"
        )
        self._set_init_progress_group_env_vars()

    def _set_init_progress_group_env_vars(self) -> None:
        # set environment variables needed for initializing torch distributed process group
        os.environ["MASTER_ADDR"] = str(self._main_address)
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        os.environ["MASTER_PORT"] = str(self._main_port)
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

    def creates_processes_externally(self) -> bool:
        """LSF creates subprocesses -- i.e. PyTorch Lightning does not need to spawn them."""
        return True

    def _read_hosts(self) -> list:
        """Read compute hosts that are a part of the compute job.

        LSF uses the Job Step Manager (JSM) to manage job steps. Job steps are executed by the JSM from "launch" nodes.
        Each job is assigned a launch node. This launch node will be the first node in the list contained in
        ``LSB_DJOB_RANKFILE``.
        """
        var = "LSB_DJOB_RANKFILE"
        try:
            rankfile = os.environ[var]
        except KeyError:
            raise ValueError("Could not find environment variable LSB_DJOB_RANKFILE")
        if not rankfile:
            raise ValueError("Environment variable LSB_DJOB_RANKFILE is empty")
        with open(rankfile) as f:
            ret = [line.strip() for line in f]
        # remove the launch node (i.e. the first node in LSB_DJOB_RANKFILE) from the list
        return ret[1:]

    @property
    def main_address(self) -> str:
        """The main address is read from an OpenMPI host rank file in the environment variable
        ``LSB_DJOB_RANKFILE``"""
        return self._main_address

    def _get_main_address(self) -> str:
        """A helper for getting the master address.

        Master address is assigned to the first node in the list of nodes used for the job.
        """
        hosts = self._read_hosts()
        return hosts[0]

    @property
    def main_port(self) -> int:
        """The main port is calculated from the LSF job ID."""
        return self._main_port

    def _get_main_port(self) -> int:
        """A helper for getting the master port.

        Use the LSF job ID so all ranks can compute the master port
        """
        # check for user-specified master port
        port = os.environ.get("MASTER_PORT")
        if not port:
            var = "LSB_JOBID"
            jobid = os.environ.get(var, None)
            if not jobid:
                raise ValueError("Could not find job id -- expected in environment variable %s" % var)
            else:
                port = int(jobid)
                # all ports should be in the 10k+ range
                port = int(port) % 1000 + 10000
            log.debug("calculated master port")
        else:
            log.debug("using externally specified master port")
        return port

    @staticmethod
    def detect() -> bool:
        """Returns ``True`` if the current process was launched using the ``jsrun`` command."""
        required_env_vars = {"LSB_JOBID", "LSB_DJOB_RANKFILE", "JSM_NAMESPACE_LOCAL_RANK", "JSM_NAMESPACE_SIZE"}
        return required_env_vars.issubset(os.environ.keys())

    def world_size(self) -> int:
        """The world size is read from the environment variable ``JSM_NAMESPACE_SIZE``."""
        return self._world_size

    def _get_world_size(self) -> int:
        """A helper function for getting the world size.

        Read this from the environment variable ``JSM_NAMESPACE_SIZE``
        """
        var = "JSM_NAMESPACE_SIZE"
        world_size = os.environ.get(var, None)
        if world_size is None:
            raise ValueError(
                f"Cannot determine local rank -- expected in {var} -- make sure you run your executable with jsrun"
            )
        return int(world_size)

    def set_world_size(self, size: int) -> None:
        log.debug("LSFEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        """The world size is read from the environment variable ``JSM_NAMESPACE_RANK``."""
        return self._global_rank

    def _get_global_rank(self) -> int:
        """A helper function for getting the global rank.

        Read this from the environment variable ``JSM_NAMESPACE_LOCAL_RANK``
        """
        var = "JSM_NAMESPACE_RANK"
        global_rank = os.environ.get(var)
        if global_rank is None:
            raise ValueError(
                "Cannot determine global rank -- expected in %s "
                "-- make sure you run your executable with jsrun" % var
            )
        return int(global_rank)

    def set_global_rank(self, rank: int) -> None:
        log.debug("LSFEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        """The local rank is read from the environment variable `JSM_NAMESPACE_LOCAL_RANK`."""
        return self._local_rank

    def _get_local_rank(self) -> int:
        """A helper function for getting the local rank.

        Read this from the environment variable ``JSM_NAMESPACE_LOCAL_RANK``
        """
        var = "JSM_NAMESPACE_LOCAL_RANK"
        local_rank = os.environ.get(var)
        if local_rank is None:
            raise ValueError(
                f"Cannot determine local rank -- expected in {var} -- make sure you run your executable with jsrun"
            )
        return int(local_rank)

    def node_rank(self) -> int:
        """The node rank is determined by the position of the current hostname in the OpenMPI host rank file stored
        in ``LSB_DJOB_RANKFILE``."""
        return self._node_rank

    def _get_node_rank(self) -> int:
        """A helper function for getting the node rank.

        Node rank is determined by the position of the current node in the hosts used in the job. This is calculated by
        reading all hosts from LSB_DJOB_RANKFILE and finding this nodes hostname in the list.
        """
        hosts = self._read_hosts()
        count = dict()
        for host in hosts:
            if host not in count:
                count[host] = len(count)
        return count[socket.gethostname()]

    def __str__(self) -> str:
        return self._rep
