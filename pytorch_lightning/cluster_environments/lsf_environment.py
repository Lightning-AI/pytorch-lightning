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
from pytorch_lightning import _logger as log
from pytorch_lightning.cluster_environments.cluster_environment import ClusterEnvironment


class LSFEnvironment(ClusterEnvironment):

    def __init__(self):
        super().__init__()

    def master_address(self):
        # figure out the root node addr
        try:
            root_node = sorted(set(x for x in open(os.environ['LSB_DJOB_HOSTFILE'], 'r')
                                   if 'batch' not in x and 'login' not in x))[0][:-1]
        except Exception:
            root_node = "127.0.0.1"

        os.environ["MASTER_ADDR"] = root_node
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        return root_node

    def master_port(self):
        # -----------------------
        # LSF JOB = PORT number
        # -----------------------
        # this way every process knows what port to use
        try:
            # use the last 4 numbers in the job id as the id
            default_port = os.environ["LSB_JOB_ID"]
            default_port = default_port[-4:]

            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000

        except Exception:
            default_port = 12910

        # -----------------------
        # PORT NUMBER = MASTER_PORT
        # -----------------------
        # in case the user passed it in
        try:
            default_port = os.environ["MASTER_PORT"]
        except Exception:
            os.environ["MASTER_PORT"] = str(default_port)

        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        return default_port

    def world_size(self):
        return self._world_size

    def local_rank(self):
        return int(os.environ['JSM_NAMESPACE_LOCAL_RANK'])
