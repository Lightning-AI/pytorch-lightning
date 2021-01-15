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
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only


def rank_zero_experiment(fn: Callable) -> Callable:
    """ Returns the real experiment on rank 0 and otherwise the DummyExperiment. """
    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)
        return get_experiment() or DummyExperiment()
    return experiment


class LightningLoggerBase(ABC):
    """
    Base class for experiment loggers.

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

    def __init__(
            self,
            agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
            agg_default_func: Callable[[Sequence[float]], float] = np.mean
    ):
        self._prev_step: int = -1
        self._metrics_to_agg: List[Dict[str, float]] = []
        self._agg_key_funcs = agg_key_funcs if agg_key_funcs else {}
        self._agg_default_func = agg_default_func

    def update_agg_funcs(
            self,
            agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
            agg_default_func: Callable[[Sequence[float]], float] = np.mean
    ):
        """
        Update aggregation methods.

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

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this logger."""

    def _aggregate_metrics(
            self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> Tuple[int, Optional[Dict[str, float]]]:
        """
        Aggregates metrics.

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
        """Aggregate accumulated metrics."""
        # compute the metrics
        if not self._metrics_to_agg:
            agg_mets = None
        elif len(self._metrics_to_agg) == 1:
            agg_mets = self._metrics_to_agg[0]
        else:
            agg_mets = merge_dicts(self._metrics_to_agg, self._agg_key_funcs, self._agg_default_func)
        return self._prev_step, agg_mets

    def _finalize_agg_metrics(self):
        """This shall be called before save/close."""
        agg_step, metrics_to_log = self._reduce_agg_metrics()
        self._metrics_to_agg = []

        if metrics_to_log is not None:
            self.log_metrics(metrics=metrics_to_log, step=agg_step)

    def agg_and_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Aggregates and records metrics.
        This method doesn't log the passed metrics instantaneously, but instead
        it aggregates them and logs only if metrics are ready to be logged.

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
        :meth:`~pytorch_lightning.loggers.base.LightningLoggerBase.agg_and_log_metrics` method.

        Args:
            metrics: Dictionary with metric names as keys and measured quantities as values
            step: Step number at which the metrics should be recorded
        """
        pass

    @staticmethod
    def _convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params

    @staticmethod
    def _sanitize_callable_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize callable params dict, e.g. ``{'a': <function_**** at 0x****>} -> {'a': 'function_****'}``.

        Args:
            params: Dictionary containing the hyperparameters

        Returns:
            dictionary with all callables sanitized
        """
        def _sanitize_callable(val):
            # Give them one chance to return a value. Don't go rabbit hole of recursive call
            if isinstance(val, Callable):
                try:
                    _val = val()
                    if isinstance(_val, Callable):
                        return val.__name__
                    return _val
                except Exception:
                    return getattr(val, "__name__", None)
            return val

        return {key: _sanitize_callable(val) for key, val in params.items()}

    @staticmethod
    def _flatten_dict(params: Dict[Any, Any], delimiter: str = '/') -> Dict[str, Any]:
        """
        Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

        Args:
            params: Dictionary containing the hyperparameters
            delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

        Returns:
            Flattened dict.

        Examples:
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 'c'}})
            {'a/b': 'c'}
            >>> LightningLoggerBase._flatten_dict({'a': {'b': 123}})
            {'a/b': 123}
            >>> LightningLoggerBase._flatten_dict({5: {'a': 123}})
            {'5/a': 123}
        """

        def _dict_generator(input_dict, prefixes=None):
            prefixes = prefixes[:] if prefixes else []
            if isinstance(input_dict, MutableMapping):
                for key, value in input_dict.items():
                    key = str(key)
                    if isinstance(value, (MutableMapping, Namespace)):
                        value = vars(value) if isinstance(value, Namespace) else value
                        for d in _dict_generator(value, prefixes + [key]):
                            yield d
                    else:
                        yield prefixes + [key, value if value is not None else str(None)]
            else:
                yield prefixes + [input_dict if input_dict is None else str(input_dict)]

        return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns params with non-primitvies converted to strings for logging.

        >>> params = {"float": 0.3,
        ...           "int": 1,
        ...           "string": "abc",
        ...           "bool": True,
        ...           "list": [1, 2, 3],
        ...           "namespace": Namespace(foo=3),
        ...           "layer": torch.nn.BatchNorm1d}
        >>> import pprint
        >>> pprint.pprint(LightningLoggerBase._sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
        {'bool': True,
         'float': 0.3,
         'int': 1,
         'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
         'list': '[1, 2, 3]',
         'namespace': 'Namespace(foo=3)',
         'string': 'abc'}
        """
        for k in params.keys():
            # convert relevant np scalars to python types first (instead of str)
            if isinstance(params[k], (np.bool_, np.integer, np.floating)):
                params[k] = params[k].item()
            elif type(params[k]) not in [bool, int, float, str, torch.Tensor]:
                params[k] = str(params[k])
        return params

    @abstractmethod
    def log_hyperparams(self, params: argparse.Namespace):
        """
        Record hyperparameters.

        Args:
            params: :class:`~argparse.Namespace` containing the hyperparameters
        """

    def log_graph(self, model: LightningModule, input_array=None) -> None:
        """
        Record model graph

        Args:
            model: lightning model
            input_array: input passes to `model.forward`
        """
        pass

    def save(self) -> None:
        """Save log data."""
        self._finalize_agg_metrics()

    def finalize(self, status: str) -> None:
        """
        Do any processing that is necessary to finalize an experiment.

        Args:
            status: Status that the experiment finished with (e.g. success, failed, aborted)
        """
        self.save()

    def close(self) -> None:
        """Do any cleanup that is necessary to close an experiment."""
        self.save()

    @property
    def save_dir(self) -> Optional[str]:
        """
        Return the root directory where experiment logs get saved, or `None` if the logger does not
        save data locally.
        """
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the experiment name."""

    @property
    @abstractmethod
    def version(self) -> Union[int, str]:
        """Return the experiment version."""

    def _add_prefix(self, metrics: Dict[str, float]):
        if self._prefix:
            metrics = {f'{self._prefix}{self.LOGGER_JOIN_CHAR}{k}': v for k, v in metrics.items()}

        return metrics


class LoggerCollection(LightningLoggerBase):
    """
    The :class:`LoggerCollection` class is used to iterate all logging actions over
    the given `logger_iterable`.

    Args:
        logger_iterable: An iterable collection of loggers
    """

    def __init__(self, logger_iterable: Iterable[LightningLoggerBase]):
        super().__init__()
        self._logger_iterable = logger_iterable

    def __getitem__(self, index: int) -> LightningLoggerBase:
        return [logger for logger in self._logger_iterable][index]

    def update_agg_funcs(
            self,
            agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
            agg_default_func: Callable[[Sequence[float]], float] = np.mean
    ):
        for logger in self._logger_iterable:
            logger.update_agg_funcs(agg_key_funcs, agg_default_func)

    @property
    def experiment(self) -> List[Any]:
        return [logger.experiment for logger in self._logger_iterable]

    def agg_and_log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for logger in self._logger_iterable:
            logger.agg_and_log_metrics(metrics, step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for logger in self._logger_iterable:
            logger.log_metrics(metrics, step)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        for logger in self._logger_iterable:
            logger.log_hyperparams(params)

    def log_graph(self, model: LightningModule, input_array=None) -> None:
        for logger in self._logger_iterable:
            logger.log_graph(model, input_array)

    def save(self) -> None:
        for logger in self._logger_iterable:
            logger.save()

    def finalize(self, status: str) -> None:
        for logger in self._logger_iterable:
            logger.finalize(status)

    def close(self) -> None:
        for logger in self._logger_iterable:
            logger.close()

    @property
    def save_dir(self) -> Optional[str]:
        # Checkpoints should be saved to default / chosen location when using multiple loggers
        return None

    @property
    def name(self) -> str:
        return '_'.join([str(logger.name) for logger in self._logger_iterable])

    @property
    def version(self) -> str:
        return '_'.join([str(logger.version) for logger in self._logger_iterable])


class DummyExperiment(object):
    """ Dummy experiment """
    def nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx):
        # enables self.logger[0].experiment.add_image
        # and self.logger.experiment[0].add_image(...)
        return self


class DummyLogger(LightningLoggerBase):
    """ Dummy logger for internal use. Is usefull if we want to disable users
        logger for a feature, but still secure that users code can run """
    def __init__(self):
        super().__init__()
        self._experiment = DummyExperiment()

    @property
    def experiment(self):
        return self._experiment

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @property
    def name(self):
        pass

    @property
    def version(self):
        pass

    def __getitem__(self, idx):
        return self


def merge_dicts(
        dicts: Sequence[Mapping],
        agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None,
        default_func: Callable[[Sequence[float]], float] = np.mean
) -> Dict:
    """
    Merge a sequence with dictionaries into one dictionary by aggregating the
    same keys with some given function.

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
    agg_key_funcs = agg_key_funcs or dict()
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
