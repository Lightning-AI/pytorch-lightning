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
from typing import Any, Callable


class _Launcher(ABC):
    r"""
    Abstract base class for all Launchers.

    Launchers are responsible for the creation and instrumentation of new processes so that the
    :class:`~pytorch_lightning.strategies.base.Strategy` can set up communication between all them.

    Subclass this class and override any of the relevant methods to provide a custom implementation depending on
    cluster environment, hardware, strategy, etc.
    """

    @property
    @abstractmethod
    def is_interactive_compatible(self) -> bool:
        """Returns whether this launcher can work in interactive environments such as Jupyter notebooks."""

    @abstractmethod
    def launch(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        """Launches the processes."""
