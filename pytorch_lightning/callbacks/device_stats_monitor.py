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
====================

Monitors and logs device stats during training.

"""
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT


class DeviceStatsMonitor(Callback):
    r"""
    Automatically monitors and logs device stats during training stage. ``DeviceStatsMonitor``
    is a special callback as it requires a ``logger`` to passed as argument to the ``Trainer``.

    Raises:
        MisconfigurationException:
            If ``Trainer`` has no logger.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import DeviceStatsMonitor
        >>> device_stats = DeviceStatsMonitor() # doctest: +SKIP
        >>> trainer = Trainer(callbacks=[device_stats]) # doctest: +SKIP
    """

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if not trainer.logger:
            raise MisconfigurationException("Cannot use DeviceStatsMonitor callback with Trainer that has no logger.")

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if not trainer.logger_connector.should_update_logs:
            return

        device_stats = trainer.accelerator.get_device_stats(pl_module.device)
        prefixed_device_stats = prefix_metrics_keys(device_stats, "on_train_batch_start")
        assert trainer.logger is not None
        trainer.logger.log_metrics(prefixed_device_stats, step=trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if not trainer.logger_connector.should_update_logs:
            return

        device_stats = trainer.accelerator.get_device_stats(pl_module.device)
        prefixed_device_stats = prefix_metrics_keys(device_stats, "on_train_batch_end")
        assert trainer.logger is not None
        trainer.logger.log_metrics(prefixed_device_stats, step=trainer.global_step)


def prefix_metrics_keys(metrics_dict: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {prefix + "." + k: v for k, v in metrics_dict.items()}
