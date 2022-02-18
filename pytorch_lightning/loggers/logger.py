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

import argparse
import functools
import operator
from abc import ABC, abstractmethod
from argparse import Namespace
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from weakref import ReferenceType

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_only


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
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

    Note:
        The `agg_key_funcs` and `agg_default_func` arguments are used only when
        one logs metrics with the :meth:`~Logger.agg_and_log_metrics` method.
    """

    def __init__(
        self,
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Callable[[Sequence[float]], float] = np.mean,
    ):
        self._prev_step: int = -1
        self._metrics_to_agg: List[Dict[str, float]] = []
        self._agg_key_funcs = agg_key_funcs if agg_key_funcs else {}
        self._agg_default_func = agg_default_func

    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        """Called after model checkpoint callback saves a new checkpoint.

        Args:
            checkpoint_callback: the model checkpoint callback instance
        """
        pass

    def update_agg_funcs(
        self,
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Callable[[Sequence[float]], float] = np.mean,
    ):
        """Update aggregation methods.

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

    def _aggregate_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> Tuple[int, Optional[Dict[str, float]]]:
        """Aggregates metrics.

        .. deprecated:: v1.6
            This method is deprecated in v1.6 and will be removed in v1.8.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded

        Returns:
            Step and aggregated metrics. The return value could be ``None``. In such case, metrics
            are added to the aggregation list, but not aggregated yet.
        """
        # if you still receiving metric from the same step, just accumulate it
        if step == self._prev_step:
            self._metrics_to_agg.append(metrics)
            return step, None

        # compute the metrics
        agg_step, agg_mets = self._reduce_agg_metrics()

        # as new step received reset accumulator
        self._metrics_to_agg = [metrics]
        self._prev_step = step
        return agg_step, agg_mets

    def _reduce_agg_metrics(self):
        """Aggregate accumulated metrics.

        See deprecation warning below.

        .. deprecated:: v1.6
            This method is deprecated in v1.6 and will be removed in v1.8.
        """
        # compute the metrics
        if not self._metrics_to_agg:
            agg_mets = None
        elif len(self._metrics_to_agg) == 1:
            agg_mets = self._metrics_to_agg[0]
        else:
            agg_mets = merge_dicts(self._metrics_to_agg, self._agg_key_funcs, self._agg_default_func)
        return self._prev_step, agg_mets

    def _finalize_agg_metrics(self):
        """This shall be called before save/close.

        See deprecation warning below.

        .. deprecated:: v1.6
            This method is deprecated in v1.6 and will be removed in v1.8.
        """
        agg_step, metrics_to_log = self._reduce_agg_metrics()
        self._metrics_to_agg = []

        if metrics_to_log is not None:
            self.log_metrics(metrics=metrics_to_log, step=agg_step)

    def agg_and_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Aggregates and records metrics. This method doesn't log the passed metrics instantaneously, but instead
        it aggregates them and logs only if metrics are ready to be logged.

        .. deprecated:: v1.6
            This method is deprecated in v1.6 and will be removed in v1.8.
            Please use `Logger.log_metrics` instead.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        agg_step, metrics_to_log = self._aggregate_metrics(metrics=metrics, step=step)

        if metrics_to_log:
            self.log_metrics(metrics=metrics_to_log, step=agg_step)

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Records metrics.
        This method logs metrics as as soon as it received them. If you want to aggregate
        metrics for one specific `step`, use the
        :meth:`~pytorch_lightning.loggers.logger.Logger.agg_and_log_metrics` method.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        pass

    @abstractmethod
    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        """Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` containing the hyperparameters
            args: Optional positional arguments, depends on the specific logger being used
            kwargs: Optional keywoard arguments, depends on the specific logger being used
        """

    def log_graph(self, model: "pl.LightningModule", input_array=None) -> None:
        """Record model graph.

        Args:
            model: lightning model
            input_array: input passes to `model.forward`
        """
        pass

    def save(self) -> None:
        """Save log data."""
        self._finalize_agg_metrics()

    def finalize(self, status: str) -> None:
        """Do any processing that is necessary to finalize an experiment.

        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()

    def close(self) -> None:
        """Do any cleanup that is necessary to close an experiment.

        See deprecation warning below.

        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.7.
            Please use `Logger.finalize` instead.
        """
        rank_zero_deprecation(
            "`Logger.close` method is deprecated in v1.5 and will be removed in v1.7."
            " Please use `Logger.finalize` instead."
        )
        self.save()

    @property
    def save_dir(self) -> Optional[str]:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return None

    @property
    def group_separator(self):
        """Return the default separator used by the logger to group the data into subfolders."""
        return "/"

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the experiment name."""

    @property
    @abstractmethod
    def version(self) -> Union[int, str]:
        """Return the experiment version."""


class LoggerCollection(Logger):
    """The :class:`LoggerCollection` class is used to iterate all logging actions over the given `logger_iterable`.

    Args:
        logger_iterable: An iterable collection of loggers
    """

    def __init__(self, logger_iterable: Iterable[Logger]):
        super().__init__()
        self._logger_iterable = logger_iterable

    def __getitem__(self, index: int) -> Logger:
        return list(self._logger_iterable)[index]

    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        for logger in self._logger_iterable:
            logger.after_save_checkpoint(checkpoint_callback)

    def update_agg_funcs(
        self,
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        agg_default_func: Callable[[Sequence[float]], float] = np.mean,
    ):
        for logger in self._logger_iterable:
            logger.update_agg_funcs(agg_key_funcs, agg_default_func)

    @property
    def experiment(self) -> List[Any]:
        """Returns a list of experiment objects for all the loggers in the logger collection."""
        return [logger.experiment for logger in self._logger_iterable]

    def agg_and_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for logger in self._logger_iterable:
            logger.agg_and_log_metrics(metrics=metrics, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for logger in self._logger_iterable:
            logger.log_metrics(metrics=metrics, step=step)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        for logger in self._logger_iterable:
            logger.log_hyperparams(params)

    def log_graph(self, model: "pl.LightningModule", input_array=None) -> None:
        for logger in self._logger_iterable:
            logger.log_graph(model, input_array)

    def log_text(self, *args, **kwargs) -> None:
        for logger in self._logger_iterable:
            logger.log_text(*args, **kwargs)

    def log_image(self, *args, **kwargs) -> None:
        for logger in self._logger_iterable:
            logger.log_image(*args, **kwargs)

    def save(self) -> None:
        for logger in self._logger_iterable:
            logger.save()

    def finalize(self, status: str) -> None:
        for logger in self._logger_iterable:
            logger.finalize(status)

    def close(self) -> None:
        """
        .. deprecated:: v1.5
            This method is deprecated in v1.5 and will be removed in v1.7.
            Please use `LoggerCollection.finalize` instead.
        """
        rank_zero_deprecation(
            "`LoggerCollection.close` method is deprecated in v1.5 and will be removed in v1.7."
            " Please use `LoggerCollection.finalize` instead."
        )
        for logger in self._logger_iterable:
            logger.close()

    @property
    def save_dir(self) -> Optional[str]:
        """Returns ``None`` as checkpoints should be saved to default / chosen location when using multiple
        loggers."""
        # Checkpoints should be saved to default / chosen location when using multiple loggers
        return None

    @property
    def name(self) -> str:
        """Returns the unique experiment names for all the loggers in the logger collection joined by an
        underscore."""
        return "_".join(dict.fromkeys(str(logger.name) for logger in self._logger_iterable))

    @property
    def version(self) -> str:
        """Returns the unique experiment versions for all the loggers in the logger collection joined by an
        underscore."""
        return "_".join(dict.fromkeys(str(logger.version) for logger in self._logger_iterable))


class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


class DummyLogger(Logger):
    """Dummy logger for internal use.

    It is useful if we want to disable user's logger for a feature, but still ensure that user code can run
    """

    def __init__(self):
        super().__init__()
        self._experiment = DummyExperiment()

    @property
    def experiment(self) -> DummyExperiment:
        """Return the experiment object associated with this logger."""
        return self._experiment

    def log_metrics(self, *args, **kwargs) -> None:
        pass

    def log_hyperparams(self, *args, **kwargs) -> None:
        pass

    @property
    def name(self) -> str:
        """Return the experiment name."""
        return ""

    @property
    def version(self) -> str:
        """Return the experiment version."""
        return ""

    def __getitem__(self, idx) -> "DummyLogger":
        # enables self.logger[0].experiment.add_image(...)
        return self

    def __iter__(self):
        # if DummyLogger is substituting a logger collection, pretend it is empty
        yield from ()


def merge_dicts(
    dicts: Sequence[Mapping],
    agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
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
    d_out = {}
    for k in keys:
        fn = agg_key_funcs.get(k)
        values_to_agg = [v for v in [d_in.get(k) for d_in in dicts] if v is not None]

        if isinstance(values_to_agg[0], dict):
            d_out[k] = merge_dicts(values_to_agg, fn, default_func)
        else:
            d_out[k] = (fn or default_func)(values_to_agg)

    return d_out


class LightningLoggerBase(Logger):
    """Base class for experiment loggers.

    Args:
        agg_key_funcs:
            Dictionary which maps a metric name to a function, which will
            aggregate the metric values for the same steps.
        agg_default_func:
            Default function to aggregate metric values. If some metric name
            is not presented in the `agg_key_funcs` dictionary, then the
            `agg_default_func` will be used for aggregation.

    Note:
        The `agg_key_funcs` and `agg_default_func` arguments are used only when
        one logs metrics with the :meth:`~LightningLoggerBase.agg_and_log_metrics` method.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        rank_zero_deprecation(
            "The `pl.loggers.base.LightningLoggerBase` is deprecated. "
            " Please use `pl.loggers.logger.Logger` instead."
        )
        super().__init__(*args, **kwargs)
