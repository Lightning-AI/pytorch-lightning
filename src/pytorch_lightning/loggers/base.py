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

from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

import pytorch_lightning.loggers.logger as logger
from pytorch_lightning.utilities.warnings import rank_zero_deprecation


def rank_zero_experiment(fn: Callable) -> Callable:
    rank_zero_deprecation(
        "The `pytorch_lightning.loggers.base.rank_zero_experiment` is deprecated in v1.7"
        " and will be removed in v1.9. Please use `pytorch_lightning.loggers.logger.rank_zero_experiment` instead."
    )
    return logger.rank_zero_experiment(fn)


class LightningLoggerBase(logger.Logger):
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

    Note:
        The `agg_key_funcs` and `agg_default_func` arguments are used only when
        one logs metrics with the :meth:`~LightningLoggerBase.agg_and_log_metrics` method.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        rank_zero_deprecation(
            "The `pytorch_lightning.loggers.base.LightningLoggerBase` is deprecated in v1.7"
            " and will be removed in v1.9. Please use `pytorch_lightning.loggers.logger.Logger` instead."
        )
        super().__init__(*args, **kwargs)


class LoggerCollection(logger.LoggerCollection):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)


class DummyExperiment(logger.DummyExperiment):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        rank_zero_deprecation(
            "The `pytorch_lightning.loggers.base.DummyExperiment` is deprecated in v1.7"
            " and will be removed in v1.9. Please use `pytorch_lightning.loggers.logger.DummyExperiment` instead."
        )
        super().__init__(*args, **kwargs)


class DummyLogger(logger.DummyLogger):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        rank_zero_deprecation(
            "The `pytorch_lightning.loggers.base.DummyLogger` is deprecated in v1.7"
            " and will be removed in v1.9. Please use `pytorch_lightning.loggers.logger.DummyLogger` instead."
        )

        super().__init__(*args, **kwargs)


def merge_dicts(
    dicts: Sequence[Mapping],
    agg_key_funcs: Optional[Mapping] = None,
    default_func: Callable[[Sequence[float]], float] = np.mean,
) -> Dict:
    rank_zero_deprecation(
        "The `pytorch_lightning.loggers.base.merge_dicts` is deprecated in v1.7"
        " and will be removed in v1.9. Please use `pytorch_lightning.loggers.logger.merge_dicts` instead."
    )
    return logger.merge_dicts(dicts=dicts, agg_key_funcs=agg_key_funcs, default_func=default_func)
