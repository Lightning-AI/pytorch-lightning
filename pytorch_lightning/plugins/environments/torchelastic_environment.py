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
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_warn


class TorchElasticEnvironment(ClusterEnvironment):

    def __init__(self):
        super().__init__()

    def master_address(self):
        if "MASTER_ADDR" not in os.environ:
            rank_zero_warn("MASTER_ADDR environment variable is not defined. Set as localhost")
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        master_address = os.environ.get('MASTER_ADDR')
        return master_address

    def master_port(self):
        if "MASTER_PORT" not in os.environ:
            rank_zero_warn("MASTER_PORT environment variable is not defined. Set as 12910")
            os.environ["MASTER_PORT"] = "12910"
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        port = os.environ.get('MASTER_PORT')
        return port

    def world_size(self):
        return os.environ.get('WORLD_SIZE')

    def local_rank(self):
        return int(os.environ['LOCAL_RANK'])

    def node_rank(self) -> int:
        return int(os.environ.get('GROUP_RANK', 0))
