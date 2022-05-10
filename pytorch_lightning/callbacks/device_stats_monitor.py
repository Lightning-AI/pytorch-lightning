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
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _PSUTIL_AVAILABLE
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT


class DeviceStatsMonitor(Callback):
    r"""
    Automatically monitors and logs device stats during training stage. ``DeviceStatsMonitor``
    is a special callback as it requires a ``logger`` to passed as argument to the ``Trainer``.

    Args:
        cpu_stats: if ``None``, it will log CPU stats only if the accelerator is CPU.
            It will raise a warning if ``psutil`` is not installed till v1.9.0.
            If ``True``, it will log CPU stats regardless of the accelerator, and it will
            raise an exception if ``psutil`` is not installed.
            If ``False``, it will not log CPU stats regardless of the accelerator.

    Raises:
        MisconfigurationException:
            If ``Trainer`` has no logger.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import DeviceStatsMonitor
        >>> device_stats = DeviceStatsMonitor() # doctest: +SKIP
        >>> trainer = Trainer(callbacks=[device_stats]) # doctest: +SKIP
    """

    def __init__(self, cpu_stats: Optional[bool] = None) -> None:
        self._cpu_stats = cpu_stats

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:
        if stage != "fit":
            return

        if not trainer.loggers:
            raise MisconfigurationException("Cannot use `DeviceStatsMonitor` callback with `Trainer(logger=False)`.")

        # warn in setup to warn once
        device = trainer.strategy.root_device
        if self._cpu_stats is None and device.type == "cpu" and not _PSUTIL_AVAILABLE:
            # TODO: raise an exception from v1.9
            rank_zero_warn(
                "`DeviceStatsMonitor` will not log CPU stats as `psutil` is not installed."
                " To install `psutil`, run `pip install psutil`."
                " It will raise an exception if `psutil` is not installed post v1.9.0."
            )
            self._cpu_stats = False

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
            from pytorch_lightning.accelerators.cpu import get_cpu_stats

            device_stats.update(get_cpu_stats())

        for logger in trainer.loggers:
            separator = logger.group_separator
            prefixed_device_stats = _prefix_metric_keys(device_stats, f"{self.__class__.__qualname__}.{key}", separator)
            logger.log_metrics(prefixed_device_stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        self._get_and_log_device_stats(trainer, "on_train_batch_start")

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self._get_and_log_device_stats(trainer, "on_train_batch_end")


def _prefix_metric_keys(metrics_dict: Dict[str, float], prefix: str, separator: str) -> Dict[str, float]:
    return {prefix + separator + k: v for k, v in metrics_dict.items()}


def prefix_metric_keys(metrics_dict: Dict[str, float], prefix: str) -> Dict[str, float]:
    rank_zero_deprecation(
        "`pytorch_lightning.callbacks.device_stats_monitor.prefix_metrics`"
        " is deprecated in v1.6 and will be removed in v1.8."
    )
    sep = ""
    return _prefix_metric_keys(metrics_dict, prefix, sep)
