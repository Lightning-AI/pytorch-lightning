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
from typing import Any, Type

from pytorch_lightning.utilities import rank_zero_deprecation


class ClusterEnvironment(ABC):
    """Specification of a cluster environment."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "ClusterEnvironment":
        # TODO: remove in 1.7
        _check_for_deprecated_methods(cls)
        return super().__new__(cls)

    @property
    @abstractmethod
    def creates_processes_externally(self) -> bool:
        """Whether the environment creates the subprocesses or not."""

    @property
    @abstractmethod
    def main_address(self) -> str:
        """The main address through which all processes connect and communicate."""

    @property
    @abstractmethod
    def main_port(self) -> int:
        """An open and configured port in the main node through which all processes communicate."""

    @staticmethod
    @abstractmethod
    def detect() -> bool:
        """Detects the environment settings corresponding to this cluster and returns ``True`` if they match."""

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


def _check_for_deprecated_methods(cls: Type[ClusterEnvironment]) -> None:
    if hasattr(cls, "master_address") and callable(cls.master_address):
        rank_zero_deprecation(
            f"`{cls.__name__}.master_address` has been deprecated in v1.6 and will be removed in v1.7."
            " Implement the property `main_address` instead (do not forget to add the `@property` decorator)."
        )
    if hasattr(cls, "master_port") and callable(cls.master_port):
        rank_zero_deprecation(
            f"`{cls.__name__}.master_port` has been deprecated in v1.6 and will be removed in v1.7."
            " Implement the property `main_port` instead (do not forget to add the `@property` decorator)."
        )
