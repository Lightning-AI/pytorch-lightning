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
"""
Model Summary
=============

Generates a summary of all layers in a :class:`~pytorch_lightning.core.lightning.LightningModule`.

The string representation of this summary prints a table with columns containing
the name, type and number of parameters for each layer.

"""
import logging
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.model_summary import _format_summary_table, summarize

log = logging.getLogger(__name__)


class ModelSummary(Callback):
    r"""
    Generates a summary of all layers in a :class:`~pytorch_lightning.core.lightning.LightningModule`.

    Args:
        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
            layer summary off. Default: 1.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import ModelSummary
        >>> trainer = Trainer(callbacks=[ModelSummary(max_depth=1)])
    """

    def __init__(self, max_depth: Optional[int] = 1):
        self._max_depth: int = max_depth
        self._summary_data: list
        self._total_parameters: int
        self._trainable_paramaters: int
        self._model_size: float

    def on_pretrain_routine_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero and self._max_depth is not None and not trainer.testing:
            model_summary = summarize(pl_module, max_depth=self._max_depth)

            self._summary_data = model_summary._get_summary_data()
            self._total_parameters = model_summary.total_parameters
            self._trainable_parameters = model_summary.trainable_parameters
            self._model_size = model_summary.model_size

            self.summarize()

    def summarize(self) -> None:
        summary_table = _format_summary_table(
            self._total_parameters, self._trainable_parameters, self._model_size, *self._summary_data
        )
        log.info("\n" + summary_table)
