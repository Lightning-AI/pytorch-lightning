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
import abc
from typing import Any, List, Optional

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class BasePredictionWriter(Callback):
    """
    Base class to implement how the predictions should be stored.

    2 hooks to override are provided:
        - write_on_batch_end: Logic to write a single batch.
        - write_on_epoch_end: Logic to write all batches.

    Args:

        write_interval: When to perform writing.
            Currently, only "step" and "epoch" are supported.

    Example::

        import torch
        import os
        from pytorch_lightning.callbacks import BasePredictionWriter

        class CustomWriter(BasePredictionWriter):

            def __init__(self, output_dir: str, write_interval: str):
                super().__init__(write_interval)
                self.output_dir

            def write_on_batch_end(
                self, trainer, pl_module: 'LightningModule', prediction: Any, batch_indices: List[int], batch: Any,
                batch_idx: int, dataloader_idx: int
            ):
                torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt")

            def write_on_epoch_end(
                self, trainer, pl_module: 'LightningModule', predictions: List[Any], batch_indices: List[Any]
            ):
                torch.save(predictions, os.path.join(self.output_dir, "predictions.pt")
    """

    write_intervals = ("batch", "epoch", "batch_and_epoch")

    @abc.abstractclassmethod
    def write_on_batch_end(
        self, trainer, pl_module: 'LightningModule', prediction: Any, batch_indices: Optional[List[int]], batch: Any,
        batch_idx: int, dataloader_idx: int
    ) -> None:
        pass

    @abc.abstractclassmethod
    def write_on_epoch_end(
        self, trainer, pl_module: 'LightningModule', predictions: List[Any], batch_indices: Optional[List[Any]]
    ) -> None:
        pass

    def __init__(self, write_interval: str):
        if write_interval is None:
            write_interval = "batch"
        if not isinstance(write_interval,
                          str) or (isinstance(write_interval, str) and write_interval not in self.write_intervals):
            raise MisconfigurationException(f"`write_interval` should be within {self.write_intervals}.")

        self._write_interval = write_interval

    def on_predict_batch_end(
        self, trainer, pl_module: 'LightningModule', outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if "batch" in self._write_interval:
            if trainer.accelerator_connector.is_distributed:
                batch_indices = trainer.predict_loop.batch_indices
            else:
                batch_indices = None
            self.write_on_batch_end(trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx)

    def on_predict_epoch_end(self, trainer, pl_module: 'LightningModule', outputs: List[Any]) -> None:
        if "epoch" in self._write_interval:
            if trainer.accelerator_connector.is_distributed:
                batch_indices = trainer.predict_loop._batches_indices
            else:
                batch_indices = None
            self.write_on_epoch_end(trainer, pl_module, trainer.predict_loop._predictions, batch_indices)
