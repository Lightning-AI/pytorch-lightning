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
import os
from typing import Any, List, Optional, Union
import torch
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class PredictionWriter(Callback):

    write_intervals = ("step", "epoch")

    def write_on_batch(self, trainer, pl_module: 'LightningModule', prediction: Any, batch_indices: List[int], batch: Any, batch_idx: int, dataloader_idx: int):
        pass

    def write_on_epoch(self, trainer, pl_module: 'LightningModule', predictions: List[Any], batch_indices: List[Any]):
        pass

    def __init__(self, write_interval: str = "step"):
        if not isinstance(write_interval, str) or (isinstance(write_interval, str) and write_interval not in self.write_intervals):
            raise MisconfigurationException(
                f"`write_interval` should be within {self.write_intervals}.on_batch_end"
            )
        
        self._write_interval = write_interval

    def on_predict_batch_end(
        self, trainer, pl_module: 'LightningModule', outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if self._write_interval == "step":
            self.write_on_batch(trainer, pl_module, outputs, trainer.predict_loop.batch_indices, batch, batch_idx, dataloader_idx)

    def on_predict_epoch_end(self, trainer, pl_module: 'LightningModule', outputs: List[Any]) -> None:
        if self._write_interval == "epoch":
            self.write_on_epoch(trainer, pl_module, trainer.predict_loop._predictions, trainer.predict_loop._batches_indices)