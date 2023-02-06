# Copyright The Lightning AI team.
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
BasePredictionWriter
====================

Aids in saving predictions
"""
from typing import Any, Optional, Sequence

from typing_extensions import Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class WriteInterval(LightningEnum):
    BATCH = "batch"
    EPOCH = "epoch"
    BATCH_AND_EPOCH = "batch_and_epoch"

    @property
    def on_batch(self) -> bool:
        return self in (self.BATCH, self.BATCH_AND_EPOCH)

    @property
    def on_epoch(self) -> bool:
        return self in (self.EPOCH, self.BATCH_AND_EPOCH)


class BasePredictionWriter(Callback):
    """Base class to implement how the predictions should be stored.

    Args:
        write_interval: When to write.

    Example::

        import torch
        from pytorch_lightning.callbacks import BasePredictionWriter

        class CustomWriter(BasePredictionWriter):

            def __init__(self, output_dir, write_interval):
                super().__init__(write_interval)
                self.output_dir = output_dir

            def write_on_batch_end(
                self, trainer, pl_module', prediction, batch_indices, batch, batch_idx, dataloader_idx
            ):
                torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))

            def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
                torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))


        pred_writer = CustomWriter(output_dir="pred_path", write_interval="epoch")
        trainer = Trainer(callbacks=[pred_writer])
        model = BoringModel()
        trainer.predict(model, return_predictions=False)

    Example::

        # multi-device inference example

        import torch
        from pytorch_lightning.callbacks import BasePredictionWriter

        class CustomWriter(BasePredictionWriter):

            def __init__(self, output_dir, write_interval):
                super().__init__(write_interval)
                self.output_dir = output_dir

            def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
                # this will create N (num processes) files in `output_dir` each containing
                # the predictions of it's respective rank
                torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

                # optionally, you can also save `batch_indices` to get the information about the data index
                # from your prediction data
                torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


        # or you can set `writer_interval="batch"` and override `write_on_batch_end` to save
        # predictions at batch level
        pred_writer = CustomWriter(output_dir="pred_path", write_interval="epoch")
        trainer = Trainer(accelerator="gpu", strategy="ddp", devices=8, callbacks=[pred_writer])
        model = BoringModel()
        trainer.predict(model, return_predictions=False)
    """

    def __init__(self, write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch") -> None:
        if write_interval not in list(WriteInterval):
            raise MisconfigurationException(f"`write_interval` should be one of {[i.value for i in WriteInterval]}.")
        self.interval = WriteInterval(write_interval)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Override with the logic to write a single batch."""
        raise NotImplementedError()

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        """Override with the logic to write all batches."""
        raise NotImplementedError()

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.interval.on_batch:
            return
        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx)

    def on_predict_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Sequence[Any]
    ) -> None:
        if not self.interval.on_epoch:
            return
        epoch_batch_indices = trainer.predict_loop.epoch_batch_indices
        self.write_on_epoch_end(trainer, pl_module, trainer.predict_loop.predictions, epoch_batch_indices)
