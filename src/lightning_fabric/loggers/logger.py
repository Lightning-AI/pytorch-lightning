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
"""Abstract base class used to build new loggers."""

from abc import ABC, abstractmethod
from argparse import Namespace
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

from torch import Tensor
from torch.nn import Module

from lightning_fabric.utilities.rank_zero import rank_zero_only


class Logger(ABC):
    """Base class for experiment loggers."""

    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        """Return the experiment name."""

    @property
    @abstractmethod
    def version(self) -> Optional[Union[int, str]]:
        """Return the experiment version."""

    @property
    def root_dir(self) -> Optional[str]:
        """Return the root directory where all versions of an experiment get saved, or `None` if the logger does
        not save data locally."""
        return None

    @property
    def log_dir(self) -> Optional[str]:
        """Return directory the current version of the experiment gets saved, or `None` if the logger does not save
        data locally."""
        return None

    @property
    def group_separator(self) -> str:
        """Return the default separator used by the logger to group the data into subfolders."""
        return "/"

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Records metrics. This method logs metrics as soon as it received them.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        pass

    @abstractmethod
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        """Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` or `Dict` containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keyword arguments, depends on the specific logger being used
        """

    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        """Record model graph.

        Args:
            model: the model with an implementation of ``forward``.
            input_array: input passes to `model.forward`
        """
        pass

    def save(self) -> None:
        """Save log data."""

    def finalize(self, status: str) -> None:
        """Do any processing that is necessary to finalize an experiment.

        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the _DummyExperiment."""

    @wraps(fn)
    def experiment(self) -> Union[Any, _DummyExperiment]:  # type: ignore[no-untyped-def]
        """
        Note:
            ``self`` is a custom logger instance. The loggers typically wrap an ``experiment`` method
            with a ``@rank_zero_experiment`` decorator.

            ``Union[Any, _DummyExperiment]`` is used because the wrapped hooks have several return
            types that are specific to the custom logger. The return type here can be considered as
            ``Union[return type of logger.experiment, _DummyExperiment]``.
        """

        @rank_zero_only
        def get_experiment() -> Callable:
            return fn(self)

        return get_experiment() or _DummyExperiment()

    return experiment


class _DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args: Any, **kw: Any) -> None:
        pass

    def __getattr__(self, _: Any) -> Callable:
        return self.nop

    def __getitem__(self, idx: int) -> "_DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        pass
