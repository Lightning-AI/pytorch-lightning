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


class LSFEnvironment(ClusterEnvironment):
    """An environment for running on clusters managed by the LSF resource manager.

    It is expected that any execution using this ClusterEnvironment was executed
    using the Job Step Manager i.e. jsrun.

    This plugin expects the following environment variables:

    LSB_JOBID
      The LSF assigned job ID

    LSB_HOSTS
      The hosts used in the job. This string is expected to have the format "batch <rank_0_host> ...."

    JSM_NAMESPACE_LOCAL_RANK
      The node local rank for the task. This environment variable is set by jsrun

    JSM_NAMESPACE_SIZE
      The world size for the task. This environment variable is set by jsrun
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
        var = "LSB_HOSTS"
        hosts = os.environ.get(var)
        if not hosts:
            raise ValueError("Could not find hosts -- expected in environment variable %s" % var)
        hosts = hosts.split()
        if len(hosts) < 2:
            raise ValueError("Cannot parse hosts from LSB_HOSTS environment variable -- "
                             "expected format \"batch <rank_0_host> ...\"")
        return hosts

    def _get_master_address(self):
        """A helper for getting the master address"""
        hosts = self._read_hosts()
        return hosts[1]

    def _get_master_port(self):
        """A helper for getting the master port

        Use the LSF job ID so all ranks can compute the master port
        """
        # check for user-specified master port
        port = os.environ.get("MASTER_PORT")
        if not port:
            var = "LSB_JOBID"
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

        Read this from the environment variable JSM_NAMESPACE_LOCAL_RANK
        """
        var = "JSM_NAMESPACE_RANK"
        global_rank = os.environ.get(var)
        if global_rank is None:
            raise ValueError("Cannot determine global rank -- expected in %s "
                             "-- make sure you run your executable with jsrun" % var)
        return int(global_rank)

    def _get_local_rank(self):
        """A helper function for getting the local rank

        Read this from the environment variable JSM_NAMESPACE_LOCAL_RANK
        """
        var = "JSM_NAMESPACE_LOCAL_RANK"
        local_rank = os.environ.get(var)
        if local_rank is None:
            raise ValueError("Cannot determine local rank -- expected in %s "
                             "-- make sure you run your executable with jsrun" % var)
        return int(local_rank)

    def _get_world_size(self):
        """A helper function for getting the world size

        Read this from the environment variable JSM_NAMESPACE_SIZE
        """
        var = "JSM_NAMESPACE_SIZE"
        world_size = os.environ.get(var)
        if world_size is None:
            raise ValueError("Cannot determine local rank -- expected in %s "
                             "-- make sure you run your executable with jsrun" % var)
        return int(world_size)

    def _get_node_rank(self):
        """A helper function for getting the node rank"""
        hosts = self._read_hosts()
        count = dict()
        for host in hosts:
            if 'batch' in host or 'login' in host:
                continue
            if host not in count:
                count[host] = len(count)
        return count[socket.gethostname()]

    def master_address(self):
        """
        Master address is read from a list of hosts contained in the environment variable *LSB_HOSTS*
        """
        return self._master_address

    def master_port(self):
        """
        Master port is calculated from the LSF job ID
        """
        return self._master_port

    def world_size(self):
        """
        World size is read from the environment variable JSM_NAMESPACE_SIZE
        """
        return self._world_size

    def local_rank(self):
        """
        World size is read from the environment variable JSM_NAMESPACE_LOCAL_RANK
        """
        return self._local_rank

    def node_rank(self):
        """
        Node rank is determined by the position of the current hostname in the list of hosts stored in LSB_HOSTS
        """
        return self._node_rank

    def global_rank(self):
        """
        World size is read from the environment variable JSM_NAMESPACE_RANK
        """
        return self._global_rank
