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
from pytorch_lightning.cluster_environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_info


class TorchElasticEnvironment(ClusterEnvironment):

    def __init__(self):
        super().__init__()

    def master_address(self):
        if "MASTER_ADDR" not in os.environ:
            rank_zero_warn(
                "MASTER_ADDR environment variable is not defined. Set as localhost"
            )
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        master_address = os.environ.get('MASTER_ADDR')
        return master_address

    def master_port(self):
        if "MASTER_PORT" not in os.environ:
            rank_zero_warn(
                "MASTER_PORT environment variable is not defined. Set as 12910"
            )
            os.environ["MASTER_PORT"] = "12910"
        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        port = os.environ.get('MASTER_PORT')
        return port

    def world_size(self):
        return os.environ.get('WORLD_SIZE')

    def local_rank(self):
        return int(os.environ['LOCAL_RANK'])

    def node_rank(self):
        # TODO: use GROUP_RANK and provide a default environment class that uses NODE_RANK
        # torchelastic uses the envvar GROUP_RANK, whereas other systems(?) use NODE_RANK.
        # otherwise use given node rank or default to node rank 0
        env_vars = ['NODE_RANK', 'GROUP_RANK']
        node_ids = [(k, os.environ.get(k, None)) for k in env_vars]
        node_ids = [(k, v) for k, v in node_ids if v is not None]
        if len(node_ids) == 0:
            return 0
        if len(node_ids) > 1:
            log.warning(f"Multiple environment variables ({node_ids}) defined for node rank. Using the first one.")
        k, rank = node_ids.pop()
        rank_zero_info(f"Using environment variable {k} for node rank ({rank}).")
        return int(rank)
