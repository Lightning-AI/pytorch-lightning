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
r"""
PredictionWriter
====================

Aids in saving predictions
"""
from abc import abstractmethod
from typing import Any, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class WriteInterval(LightningEnum):
    BATCH = "batch"
    EPOCH = "epoch"
    BATCH_AND_EPOCH = "batch_and_epoch"

    @property
    def on_batch(self) -> bool:
        return self in (self.STEP, self.STEP_AND_EPOCH)

    @property
    def on_epoch(self) -> bool:
        return self in (self.EPOCH, self.BATCH_AND_EPOCH)


class BasePredictionWriter(Callback):
    """
    Base class to implement how the predictions should be stored.

    Args:
        write_interval: When to write.

    Example::

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

    def __init__(self, write_interval: str = "batch") -> None:
        if write_interval not in list(WriteInterval):
            raise MisconfigurationException(f"`write_interval` should one of {list(WriteInterval)}.")
        self.interval = WriteInterval(write_interval)

    @abstractmethod
    def write_on_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        prediction: Any,
        batch_indices: Optional[List[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Override with logic to write a single batch."""

    @abstractmethod
    def write_on_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        predictions: List[Any],
        batch_indices: Optional[List[Any]],
    ) -> None:
        """Override with logic to write all batches."""

    def on_predict_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.interval.on_batch:
            return
        batch_indices = trainer.predict_loop.batch_indices if trainer.accelerator_connector.is_distributed else None
        self.write_on_batch_end(trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx)

    def on_predict_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs: List[Any]) -> None:
        if not self.interval.on_epoch:
            return
        batch_indices = trainer.predict_loop.batch_indices if trainer.accelerator_connector.is_distributed else None
        self.write_on_epoch_end(trainer, pl_module, trainer.predict_loop._predictions, batch_indices)
