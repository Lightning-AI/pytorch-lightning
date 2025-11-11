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
import os

from typing_extensions import override

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.imports import _raise_enterprise_not_available

log = logging.getLogger(__name__)


class LSFEnvironment(ClusterEnvironment):
    """An environment for running on clusters managed by the LSF resource manager.

    It is expected that any execution using this ClusterEnvironment was executed
    using the Job Step Manager i.e. ``jsrun``.

    This plugin expects the following environment variables:

    ``LSB_JOBID``
      The LSF assigned job ID

    ``LSB_DJOB_RANKFILE``
      The OpenMPI compatible rank file for the LSF job

    ``JSM_NAMESPACE_LOCAL_RANK``
      The node local rank for the task. This environment variable is set by ``jsrun``

    ``JSM_NAMESPACE_SIZE``
      The world size for the task. This environment variable is set by ``jsrun``

    ``JSM_NAMESPACE_RANK``
      The global rank for the task. This environment variable is set by ``jsrun``

    """

    def __init__(self) -> None:
        super().__init__()

        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.environments.lsf import (
            LSFEnvironment as EnterpriseLSFEnvironment,
        )

        self.lsf_impl = EnterpriseLSFEnvironment()

    @property
    @override
    def creates_processes_externally(self) -> bool:
        """LSF creates subprocesses, i.e., PyTorch Lightning does not need to spawn them."""
        return self.lsf_impl.creates_processes_externally

    @property
    @override
    def main_address(self) -> str:
        """The main address is read from an OpenMPI host rank file in the environment variable
        ``LSB_DJOB_RANKFILE``."""
        return self.lsf_impl.main_address

    @property
    @override
    def main_port(self) -> int:
        """The main port is calculated from the LSF job ID."""
        return self.lsf_impl.main_port

    @staticmethod
    @override
    def detect() -> bool:
        """Returns ``True`` if the current process was launched using the ``jsrun`` command."""
        required_env_vars = {"LSB_JOBID", "LSB_DJOB_RANKFILE", "JSM_NAMESPACE_LOCAL_RANK", "JSM_NAMESPACE_SIZE"}
        return required_env_vars.issubset(os.environ.keys())

    @override
    def world_size(self) -> int:
        """The world size is read from the environment variable ``JSM_NAMESPACE_SIZE``."""
        return self.lsf_impl.world_size()

    @override
    def set_world_size(self, size: int) -> None:
        return self.lsf_impl.set_world_size(size)

    @override
    def global_rank(self) -> int:
        """The world size is read from the environment variable ``JSM_NAMESPACE_RANK``."""
        return self.lsf_impl.global_rank()

    @override
    def set_global_rank(self, rank: int) -> None:
        return self.lsf_impl.set_global_rank(rank)

    @override
    def local_rank(self) -> int:
        """The local rank is read from the environment variable `JSM_NAMESPACE_LOCAL_RANK`."""
        return self.lsf_impl.local_rank()

    @override
    def node_rank(self) -> int:
        """The node rank is determined by the position of the current hostname in the OpenMPI host rank file stored in
        ``LSB_DJOB_RANKFILE``."""
        return self.lsf_impl.node_rank()
