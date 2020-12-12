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
import re
import socket
import warnings
from pytorch_lightning import _logger as log
from pytorch_lightning.cluster_environments.cluster_environment import ClusterEnvironment


class SLURMEnvironment(ClusterEnvironment):
    """An environment for running on clusters managed by the LSF resource manager.

    It is expected that any execution using this ClusterEnvironment was executed
    using the srun

    This plugin expects the following environment variables:

    SLURM_JOB_ID
      The Slurm assigned job ID

    SLURM_NODELIST
      The hosts used in the job. This string is expected to have the format "<rank_0_host> ...."

    SLURM_LOCALID
      The node local rank for the task.

    SLURM_PROCID
      The MPI rank or relative process ID

    SLURM_STEP_NUM_TASKS
      The world size for the job. This environment variable is set by srun
    """

    def __init__(self):
        self._master_address = self._get_master_address()
        self._master_port = self._get_master_port()
        self._local_rank = self._get_local_rank()
        self._global_rank = self._get_global_rank()
        self._world_size = self._get_world_size()
        self._node_rank = self._get_node_rank()

        # set environment variables needed for initializing torch distributed process group
        os.environ["MASTER_ADDR"] = str(self._master_address)
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        os.environ["MASTER_PORT"] = str(self._master_port)
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

    def _read_hosts(self):
        var = "SLURM_NODELIST"
        hosts = os.environ.get(var)
        if not hosts:
            raise ValueError("Could not find hosts -- expected in environment variable %s" % var)
        hosts = hosts.split()
        return hosts

    def _get_master_address(self):
        """A helper for getting the master address"""
        hosts = self._read_hosts()
        return hosts[0]

    def _get_master_port(self):
        """A helper for getting the master port

        Use the Slurm job ID so all ranks can compute the master port
        """
        # check for user-specified master port
        port = os.environ.get("MASTER_PORT")
        if not port:
            var = "SLURM_JOB_ID"
            jobid = os.environ.get(var)
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

    def _get_global_rank(self):
        """A helper function for getting the global rank

        Read this from the environment variable SLURM_PROCID
        """
        var = "SLURM_PROCID"
        global_rank = os.environ.get(var)
        if global_rank is None:
            raise ValueError("Cannot determine global rank -- expected in %s " % var)
        return int(global_rank)

    def _get_local_rank(self):
        """A helper function for getting the local rank

        Read this from the environment variable SLURM_LOCALID
        """
        var = "SLURM_LOCALID"
        local_rank = os.environ.get(var)
        if local_rank is None:
            raise ValueError("Cannot determine local rank -- expected in %s " % var)
        return int(local_rank)

    def _get_world_size(self):
        """A helper function for getting the world size

        Read this from the environment variable SLURM_STEP_NUM_TASKS
        """
        var = "SLURM_STEP_NUM_TASKS"
        world_size = os.environ.get(var)
        if world_size is None:
            raise ValueError("Cannot determine world size -- expected in %s "
                             "-- make sure you run your executable with srun" % var)
        return int(world_size)

    def _get_node_rank(self):
        """A helper function for getting the node rank

        Read this from the environment variable SLURM_NODEID
        """
        var = "SLURM_NODEID"
        local_rank = os.environ.get(var)
        if local_rank is None:
            raise ValueError("Cannot determine node rank -- expected in %s " % var)
        return int(local_rank)

    def master_address(self):
        """
        Master address is read from a list of hosts contained in the environment variable *SLURM_NODELIST*
        """
        return self._master_address

    def master_port(self):
        """
        Master port is calculated from the Slurm job ID
        """
        return self._master_port

    def world_size(self):
        """
        World size is read from the environment variable SLURM_STEP_NUM_TASKS
        """
        return self._world_size

    def local_rank(self):
        """
        World size is read from the environment variable SLURM_LOCALID
        """
        return self._local_rank

    def node_rank(self):
        """
        Node rank is read from the environment variable SLURM_NODEID
        """
        return self._node_rank

    def global_rank(self):
        """
        World size is read from the environment variable SLURM_PROCID
        """
        return self._global_rank
