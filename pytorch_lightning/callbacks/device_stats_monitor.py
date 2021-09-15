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
Device Stats Monitor
=================

Monitors and logs device stats during training.

"""
import time
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8


class DeviceStatsMonitor(Callback):
    r"""
    Automatically monitors and logs device stats during training stage. ``DeviceStatsMonitor``
    is a callback and in order to use it you need to assign a logger in the ``Trainer``.

    Raises:
        MisconfigurationException:
            If ``Trainer`` has no logger.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import DeviceStatsMonitor
        >>> device_stats = DeviceStatsMonitor() # doctest: +SKIP
        >>> trainer = Trainer(callbacks=[device_stats]) # doctest: +SKIP

    """

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if not trainer.logger:
            raise MisconfigurationException("Cannot use DeviceStatsMonitor callback with Trainer that has no logger.")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if isinstance(trainer.accelerator, pl.accelerators.GPUAccelerator) and not _TORCH_GREATER_EQUAL_1_8:
            self._snap_intra_step_time = time.time()

        logs = trainer.accelerator.get_device_stats()

        if (
            isinstance(trainer.accelerator, pl.accelerators.GPUAccelerator)
            and not _TORCH_GREATER_EQUAL_1_8
            and self._snap_inter_step_time
        ):
            # First log at beginning of second step
            logs["batch_time/inter_step (ms)"] = (time.time() - self._snap_inter_step_time) * 1000

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if isinstance(trainer.accelerator, pl.accelerators.GPUAccelerator) and not _TORCH_GREATER_EQUAL_1_8:
            self._snap_inter_step_time = time.time()

        logs = trainer.accelerator.get_device_stats()

        if (
            isinstance(trainer.accelerator, pl.accelerators.GPUAccelerator)
            and not _TORCH_GREATER_EQUAL_1_8
            and self._log_stats.intra_step_time
            and self._snap_intra_step_time
        ):
            logs["batch_time/intra_step (ms)"] = (time.time() - self._snap_intra_step_time) * 1000

        trainer.logger.log_metrics(logs, step=trainer.global_step)
