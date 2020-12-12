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
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.cluster_environments.cluster_environment import ClusterEnvironment


class TorchElasticEnvironment(ClusterEnvironment):
    """An environment for running in an environment managed by Torch Elastic

    This ClusterEnvironment expects that it was invoked from within a job
    started with the Elastic Launcher.

    This plugin expects the following environment variables:

    MASTER_ADDR
      fqdn of the host that is running worker with rank 0

    MASTER_PORT
      port on the MASTER_ADDR that can be used to host the tcp c10d store

    WORLD_SIZE
      total number of workers in the job

    GROUP_RANK
      rank of the worker group

    RANK
      rank of the worker within a worker group

    LOCAL_RANK
       rank of the worker within a local worker group

    See `Elastic Launch <https://pytorch.org/elastic/latest/distributed.html>` for more details.
    """


    def _read_required(self, envar, target):
        """A helper for reading required environment variables"""
        ret = os.environ.get(envar)
        if ret is None:
            raise ValueError("Could not find %s -- expected in environment variable %s" % (target, envar))
        return ret

    def __init__(self):
        self._world_size = self._read_required('WORLD_SIZE', 'world size')
        self._local_rank = self._read_required('LOCAL_RANK', 'local rank')
        self._node_rank = self._read_required('GROUP_RANK', 'node rank')
        self._global_rank = self._read_required('RANK', 'global rank')
        self._master_address = self._get_master_address()
        self._master_port = self._get_master_port()

    def _get_master_address(self):
        """A helper for reading MASTER_ADDR environment variable

        If not MASTER_POR is not found, returns 127.0.0.1
        """
        if "MASTER_ADDR" not in os.environ:
            rank_zero_warn(
                "MASTER_ADDR environment variable is not defined. Set as localhost"
            )
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        master_address = os.environ.get('MASTER_ADDR')
        return master_address

    def _get_master_port(self):
        """A helper for reading MASTER_PORT environment variable

        If not MASTER_POR is not found, returns 12910
        """
        if "MASTER_PORT" not in os.environ:
            rank_zero_warn(
                "MASTER_PORT environment variable is not defined. Set as 12910"
            )
            os.environ["MASTER_PORT"] = "12910"
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        port = os.environ.get('MASTER_PORT')
        return port

    def master_address(self):
        """Read from environment variable MASTER_ADDR"""
        return self._master_address

    def master_port(self):
        """Read from environment variable MASTER_PORT"""
        return self._master_port

    def world_size(self):
        """Read from environment variable WORLD_SIZE"""
        return self._world_size

    def local_rank(self):
        """Read from environment variable LOCAL_RANK"""
        return self._local_rank

    def node_rank(self):
        """Read from environment variable GROUP_RANK"""
        return self._node_rank

    def global_rank(self):
        """Read from environment variable RANK"""
        return self._global_rank
