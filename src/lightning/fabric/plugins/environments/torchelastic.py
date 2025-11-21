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

import torch.distributed
from typing_extensions import override

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.imports import _raise_enterprise_not_available

log = logging.getLogger(__name__)


class TorchElasticEnvironment(ClusterEnvironment):
    """Environment for fault-tolerant and elastic training with `torchelastic <https://pytorch.org/elastic/>`_"""

    def __init__(self) -> None:
        super().__init__()
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.environments.torchelastic import (
            TorchElasticEnvironment as EnterpriseTorchElasticEnvironment,
        )

        self.torchelastic_impl = EnterpriseTorchElasticEnvironment()

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return self.torchelastic_impl.creates_processes_externally

    @property
    @override
    def main_address(self) -> str:
        return self.torchelastic_impl.main_address

    @property
    @override
    def main_port(self) -> int:
        return self.torchelastic_impl.main_port

    @staticmethod
    @override
    def detect() -> bool:
        """Returns ``True`` if the current process was launched using the torchelastic command."""
        # if not available (for example on MacOS), `is_torchelastic_launched` is not defined
        return torch.distributed.is_available() and torch.distributed.is_torchelastic_launched()

    @override
    def world_size(self) -> int:
        return self.torchelastic_impl.world_size()

    @override
    def set_world_size(self, size: int) -> None:
        return self.torchelastic_impl.set_world_size(size)

    @override
    def global_rank(self) -> int:
        return self.torchelastic_impl.global_rank()

    @override
    def set_global_rank(self, rank: int) -> None:
        return self.torchelastic_impl.set_global_rank(rank)

    @override
    def local_rank(self) -> int:
        return self.torchelastic_impl.local_rank()

    @override
    def node_rank(self) -> int:
        return self.torchelastic_impl.node_rank()

    @override
    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        return self.torchelastic_impl.validate_settings(num_devices, num_nodes)
