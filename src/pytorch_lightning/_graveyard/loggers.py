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

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger


class LoggerCollection:
    # TODO: Remove in v2.0.0
    def __init__(self, _: Any):
        raise NotImplementedError(
            "`LoggerCollection` was deprecated in v1.6 and removed as of v1.8. Directly pass a list of loggers"
            " to the `Trainer` and access the list via the `trainer.loggers` attribute."
        )


def _update_agg_funcs(logger: Logger, *__: Any, **___: Any) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        f"`{type(logger).__name__}.update_agg_funcs` was deprecated in v1.6 and is no longer supported as of v1.8."
    )


def _agg_and_log_metrics(logger: Logger, *__: Any, **___: Any) -> None:
    # TODO: Remove in v2.0.0
    raise NotImplementedError(
        f"`{type(logger).__name__}.agg_and_log_metrics` was deprecated in v1.6 and is no longer supported as of v1.8."
    )


# Methods
Logger.update_agg_funcs = _update_agg_funcs
Logger.agg_and_log_metrics = _agg_and_log_metrics

# Classes
pl.loggers.logger.LoggerCollection = LoggerCollection
pl.loggers.base.LoggerCollection = LoggerCollection
