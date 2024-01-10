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

from lightning.fabric.accelerators.xla import _XLA_AVAILABLE, _XLA_GREATER_EQUAL_2_1, XLAAccelerator
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment

log = logging.getLogger(__name__)


class XLAEnvironment(ClusterEnvironment):
    """Cluster environment for training on a TPU Pod with the `PyTorch/XLA <https://pytorch.org/xla>`_ library.

    A list of environment variables set by XLA can be found
    `here <https://github.com/pytorch/xla/blob/master/torch_xla/core/xla_env_vars.py>`_.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(*args, **kwargs)

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return False

    @property
    @override
    def main_address(self) -> str:
        # unused by lightning
        raise NotImplementedError

    @property
    @override
    def main_port(self) -> int:
        # unused by lightning
        raise NotImplementedError

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
        import torch_xla.core.xla_model as xm

        return xm.xrt_world_size()

    @override
    def set_world_size(self, size: int) -> None:
        log.debug("XLAEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    @override
    @functools.lru_cache(maxsize=1)
    def global_rank(self) -> int:
        """The rank (index) of the currently running process across all host and devices.

        The output is cached for performance.

        """
        import torch_xla.core.xla_model as xm

        return xm.get_ordinal()

    @override
    def set_global_rank(self, rank: int) -> None:
        log.debug("XLAEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    @override
    @functools.lru_cache(maxsize=1)
    def local_rank(self) -> int:
        """The rank (index) of the currently running process inside of the current host.

        The output is cached for performance.

        """
        import torch_xla.core.xla_model as xm

        return xm.get_local_ordinal()

    @override
    @functools.lru_cache(maxsize=1)
    def node_rank(self) -> int:
        """The rank (index) of the host on which the current process runs.

        The output is cached for performance.

        """
        if _XLA_GREATER_EQUAL_2_1:
            from torch_xla import runtime as xr

            return xr.host_index()
        import torch_xla.core.xla_env_vars as xenv
        from torch_xla.utils.utils import getenv_as

        return getenv_as(xenv.HOST_ORDINAL, int, 0)
