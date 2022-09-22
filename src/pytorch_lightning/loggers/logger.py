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
"""Abstract base class used to build new loggers."""


import functools
import operator
from abc import ABC, abstractmethod
from argparse import Namespace
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union
from weakref import ReferenceType

import numpy as np
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_only


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self) -> Union[Any, DummyExperiment]:  # type: ignore[no-untyped-def]
        """
        Note:
            ``self`` is a custom logger instance. The loggers typically wrap an ``experiment`` method
            with a ``@rank_zero_experiment`` decorator. An exception is that ``loggers.neptune`` wraps
            ``experiment`` and ``run`` with rank_zero_experiment.

            ``Union[Any, DummyExperiment]`` is used because the wrapped hooks have several return
            types that are specific to the custom logger. The return type here can be considered as
            ``Union[return type of logger.experiment, DummyExperiment]``.
        """

        @rank_zero_only
        def get_experiment() -> Callable:
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class Logger(ABC):
    """Base class for experiment loggers.

    Args:
        agg_key_funcs:
            Dictionary which maps a metric name to a function, which will
            aggregate the metric values for the same steps.
        agg_default_func:
            Default function to aggregate metric values. If some metric name
            is not presented in the `agg_key_funcs` dictionary, then the
            `agg_default_func` will be used for aggregation.

        .. deprecated:: v1.6
            The parameters `agg_key_funcs` and `agg_default_func` are deprecated
            in v1.6 and will be removed in v1.8.
    """

    def __init__(
        self,
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Optional[Callable[[Sequence[float]], float]] = None,
    ):
        self._prev_step: int = -1
        self._metrics_to_agg: List[Dict[str, float]] = []
        if agg_key_funcs:
            self._agg_key_funcs = agg_key_funcs
            rank_zero_deprecation(
                "The `agg_key_funcs` parameter for `Logger` was deprecated in v1.6" " and will be removed in v1.8."
            )
        else:
            self._agg_key_funcs = {}
        if agg_default_func:
            self._agg_default_func = agg_default_func
            rank_zero_deprecation(
                "The `agg_default_func` parameter for `Logger` was deprecated in v1.6" " and will be removed in v1.8."
            )
        else:
            self._agg_default_func = np.mean

    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[Checkpoint]") -> None:
        """Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance
        """
        pass

    def update_agg_funcs(
        self,
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Callable[[Sequence[float]], float] = np.mean,
    ) -> None:
        """Update aggregation methods.

        .. deprecated:: v1.6
            `update_agg_funcs` is deprecated in v1.6 and will be removed in v1.8.

        Args:
            agg_key_funcs:
                Dictionary which maps a metric name to a function, which will
                aggregate the metric values for the same steps.
            agg_default_func:
                Default function to aggregate metric values. If some metric name
                is not presented in the `agg_key_funcs` dictionary, then the
                `agg_default_func` will be used for aggregation.
        """
        if agg_key_funcs:
            self._agg_key_funcs.update(agg_key_funcs)
        if agg_default_func:
            self._agg_default_func = agg_default_func
        rank_zero_deprecation("`Logger.update_agg_funcs` was deprecated in v1.6 and will be removed in v1.8.")

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Records metrics.
        This method logs metrics as as soon as it received them.

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

    def log_graph(self, model: "pl.LightningModule", input_array: Optional[Tensor] = None) -> None:
        """Record model graph.

        Args:
            model: lightning model
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

    @property
    def save_dir(self) -> Optional[str]:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return None

    @property
    def group_separator(self) -> str:
        """Return the default separator used by the logger to group the data into subfolders."""
        return "/"

    @property
    @abstractmethod
    def name(self) -> Optional[str]:
        """Return the experiment name."""

    @property
    @abstractmethod
    def version(self) -> Optional[Union[int, str]]:
        """Return the experiment version."""


class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args: Any, **kw: Any) -> None:
        pass

    def __getattr__(self, _: Any) -> Callable:
        return self.nop

    def __getitem__(self, idx: int) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        pass


class DummyLogger(Logger):
    """Dummy logger for internal use.

    It is useful if we want to disable user's logger for a feature, but still ensure that user code can run
    """

    def __init__(self) -> None:
        super().__init__()
        self._experiment = DummyExperiment()

    @property
    def experiment(self) -> DummyExperiment:
        """Return the experiment object associated with this logger."""
        return self._experiment

    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        pass

    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return ""

    @property
    def version(self) -> str:
        """Return the experiment version."""
        return ""

    def __getitem__(self, idx: int) -> "DummyLogger":
        # enables self.logger[0].experiment.add_image(...)
        return self

    def __getattr__(self, name: str) -> Callable:
        """Allows the DummyLogger to be called with arbitrary methods, to avoid AttributeErrors."""

        def method(*args: Any, **kwargs: Any) -> None:
            return None

        return method


def merge_dicts(
    dicts: Sequence[Mapping],
    agg_key_funcs: Optional[Mapping] = None,
    default_func: Callable[[Sequence[float]], float] = np.mean,
) -> Dict:
    """Merge a sequence with dictionaries into one dictionary by aggregating the same keys with some given
    function.

    Args:
        dicts:
            Sequence of dictionaries to be merged.
        agg_key_funcs:
            Mapping from key name to function. This function will aggregate a
            list of values, obtained from the same key of all dictionaries.
            If some key has no specified aggregation function, the default one
            will be used. Default is: ``None`` (all keys will be aggregated by the
            default function).
        default_func:
            Default function to aggregate keys, which are not presented in the
            `agg_key_funcs` map.

    Returns:
        Dictionary with merged values.

    Examples:
        >>> import pprint
        >>> d1 = {'a': 1.7, 'b': 2.0, 'c': 1, 'd': {'d1': 1, 'd3': 3}}
        >>> d2 = {'a': 1.1, 'b': 2.2, 'v': 1, 'd': {'d1': 2, 'd2': 3}}
        >>> d3 = {'a': 1.1, 'v': 2.3, 'd': {'d3': 3, 'd4': {'d5': 1}}}
        >>> dflt_func = min
        >>> agg_funcs = {'a': np.mean, 'v': max, 'd': {'d1': sum}}
        >>> pprint.pprint(merge_dicts([d1, d2, d3], agg_funcs, dflt_func))
        {'a': 1.3,
         'b': 2.0,
         'c': 1,
         'd': {'d1': 3, 'd2': 3, 'd3': 3, 'd4': {'d5': 1}},
         'v': 2.3}
    """
    agg_key_funcs = agg_key_funcs or {}
    keys = list(functools.reduce(operator.or_, [set(d.keys()) for d in dicts]))
    d_out: Dict = defaultdict(dict)
    for k in keys:
        fn = agg_key_funcs.get(k)
        values_to_agg = [v for v in [d_in.get(k) for d_in in dicts] if v is not None]

        if isinstance(values_to_agg[0], dict):
            d_out[k] = merge_dicts(values_to_agg, fn, default_func)
        else:
            d_out[k] = (fn or default_func)(values_to_agg)

    return dict(d_out)
