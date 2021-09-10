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
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.model_summary import summarize


class ModelSummary(Callback):
    def __init__(self, max_depth: Optional[int] = 1):
        self._max_depth = max_depth

    def on_pretrain_routine_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero and self._max_depth is not None and not trainer.testing:
            summarize(pl_module, max_depth=self._max_depth)
