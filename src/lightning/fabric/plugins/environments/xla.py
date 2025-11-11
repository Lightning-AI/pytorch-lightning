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
import functools
import logging
from typing import Any

from typing_extensions import override

from lightning.fabric.accelerators.xla import XLAAccelerator
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.imports import _raise_enterprise_not_available

log = logging.getLogger(__name__)


class XLAEnvironment(ClusterEnvironment):
    """Cluster environment for training on a TPU Pod with the `PyTorch/XLA <https://pytorch.org/xla>`_ library.

    A list of environment variables set by XLA can be found
    `here <https://github.com/pytorch/xla/blob/master/torch_xla/core/xla_env_vars.py>`_.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.environments.xla import (
            XLAEnvironment as EnterpriseXLAEnvironment,
        )

        self.xla_impl = EnterpriseXLAEnvironment(*args, **kwargs)

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return self.xla_impl.creates_processes_externally

    @property
    @override
    def main_address(self) -> str:
        return self.xla_impl.main_address

    @property
    @override
    def main_port(self) -> int:
        return self.xla_impl.main_port

    @staticmethod
    @override
    def detect() -> bool:
        return XLAAccelerator.is_available()

    @override
    @functools.lru_cache(maxsize=1)
    def world_size(self) -> int:
        """The number of processes across all devices and hosts.

        The output is cached for performance.

        """
        return self.xla_impl.world_size()

    @override
    def set_world_size(self, size: int) -> None:
        return self.xla_impl.set_world_size(size)

    @override
    @functools.lru_cache(maxsize=1)
    def global_rank(self) -> int:
        """The rank (index) of the currently running process across all host and devices.

        The output is cached for performance.

        """
        return self.xla_impl.global_rank()

    @override
    def set_global_rank(self, rank: int) -> None:
        return self.xla_impl.set_global_rank(rank)

    @override
    @functools.lru_cache(maxsize=1)
    def local_rank(self) -> int:
        """The rank (index) of the currently running process inside of the current host.

        The output is cached for performance.

        """
        return self.xla_impl.local_rank()

    @override
    @functools.lru_cache(maxsize=1)
    def node_rank(self) -> int:
        """The rank (index) of the host on which the current process runs.

        The output is cached for performance.

        """
        return self.xla_impl.node_rank()
