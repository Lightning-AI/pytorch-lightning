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
"""
Device Stats Monitor
====================

Monitors and logs device stats during training.

"""

from typing import Any, Dict, Optional

from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.accelerators.cpu import _PSUTIL_AVAILABLE
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT


class DeviceStatsMonitor(Callback):
    r"""Automatically monitors and logs device stats during training, validation and testing stage.
    ``DeviceStatsMonitor`` is a special callback as it requires a ``logger`` to passed as argument to the ``Trainer``.

    Args:
        cpu_stats: if ``None``, it will log CPU stats only if the accelerator is CPU.
            If ``True``, it will log CPU stats regardless of the accelerator.
            If ``False``, it will not log CPU stats regardless of the accelerator.

    Raises:
        MisconfigurationException:
            If ``Trainer`` has no logger.
        ModuleNotFoundError:
            If ``psutil`` is not installed and CPU stats are monitored.

    Example::

        from lightning import Trainer
        from lightning.pytorch.callbacks import DeviceStatsMonitor
        device_stats = DeviceStatsMonitor()
        trainer = Trainer(callbacks=[device_stats])

    """

    def __init__(self, cpu_stats: Optional[bool] = None) -> None:
        self._cpu_stats = cpu_stats

    @override
    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str,
    ) -> None:
        if stage != "fit":
            return

        if not trainer.loggers:
            raise MisconfigurationException("Cannot use `DeviceStatsMonitor` callback with `Trainer(logger=False)`.")

        # warn in setup to warn once
        device = trainer.strategy.root_device
        if self._cpu_stats is None and device.type == "cpu" and not _PSUTIL_AVAILABLE:
            raise ModuleNotFoundError(
                f"`DeviceStatsMonitor` cannot log CPU stats as `psutil` is not installed. {str(_PSUTIL_AVAILABLE)} "
            )

    def _get_and_log_device_stats(self, trainer: "pl.Trainer", key: str) -> None:
        if not trainer._logger_connector.should_update_logs:
            return

        device = trainer.strategy.root_device
        if self._cpu_stats is False and device.type == "cpu":
            # cpu stats are disabled
            return

        device_stats = trainer.accelerator.get_device_stats(device)

        if self._cpu_stats and device.type != "cpu":
            # Don't query CPU stats twice if CPU is accelerator
            from lightning.pytorch.accelerators.cpu import get_cpu_stats

            device_stats.update(get_cpu_stats())

        for logger in trainer.loggers:
            separator = logger.group_separator
            prefixed_device_stats = _prefix_metric_keys(device_stats, f"{self.__class__.__qualname__}.{key}", separator)
            logger.log_metrics(prefixed_device_stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    @override
    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        self._get_and_log_device_stats(trainer, "on_train_batch_start")

    @override
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self._get_and_log_device_stats(trainer, "on_train_batch_end")

    @override
    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, "on_validation_batch_start")

    @override
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, "on_validation_batch_end")

    @override
    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, "on_test_batch_start")

    @override
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, "on_test_batch_end")


def _prefix_metric_keys(metrics_dict: Dict[str, float], prefix: str, separator: str) -> Dict[str, float]:
    return {prefix + separator + k: v for k, v in metrics_dict.items()}
