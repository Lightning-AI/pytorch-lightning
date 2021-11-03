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
from abc import ABC, abstractmethod

from pytorch_lightning.utilities import rank_zero_deprecation


class ClusterEnvironment(ABC):
    """Specification of a cluster environment."""

    @property
    @abstractmethod
    def creates_processes_externally(self) -> bool:
        """Whether the environment creates the subprocesses or not."""

    def creates_children(self) -> bool:
        """Whether the environment creates the subprocesses or not.

        .. deprecated:: v1.5
            This method was deprecated in v1.5 and will be removed in v1.6. Use the property
            :attr:`creates_processes_externally` instead.
        """
        rank_zero_deprecation(
            f"`{self.__class__.__name__}.creates_children()` was deprecated in v1.5 and will be removed in v1.6."
            " Use the property :attr:`creates_processes_externally` instead."
        )
        return self.creates_processes_externally

    @property
    @abstractmethod
    def main_port(self) -> int:
        """An open and configured port in the main node through which all processes communicate."""

    @abstractmethod
    def world_size(self) -> int:
        """The number of processes across all devices and nodes."""

    @abstractmethod
    def set_world_size(self, size: int) -> None:
        pass

    @abstractmethod
    def global_rank(self) -> int:
        """The rank (index) of the currently running process across all nodes and devices."""

    @abstractmethod
    def set_global_rank(self, rank: int) -> None:
        pass

    @abstractmethod
    def local_rank(self) -> int:
        """The rank (index) of the currently running process inside of the current node."""

    @abstractmethod
    def node_rank(self) -> int:
        """The rank (index) of the node on which the current process runs."""

    def teardown(self) -> None:
        """Clean up any state set after execution finishes."""
        pass
