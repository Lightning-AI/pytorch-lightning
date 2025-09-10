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

from typing import Any, Optional

from typing_extensions import override

import lightning.pytorch as pl
from lightning.pytorch.accelerators.cpu import _PSUTIL_AVAILABLE
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT


class DeviceStatsMonitor(Callback):
    r"""Automatically monitors and logs device stats during training, validation and testing stage.
    ``DeviceStatsMonitor`` is a special callback as it requires a ``logger`` to passed as argument to the ``Trainer``.

    **Logged Metrics**

    Logs device statistics with keys prefixed as ``DeviceStatsMonitor.{hook_name}/{base_metric_name}``.
    The actual metrics depend on the active accelerator and the ``cpu_stats`` flag. Below are an overview of the
    possible available metrics and their meaning.

    - CPU (via ``psutil``)

        - ``cpu_percent`` — System-wide CPU utilization (%)
        - ``cpu_vm_percent`` — System-wide virtual memory (RAM) utilization (%)
        - ``cpu_swap_percent`` — System-wide swap memory utilization (%)

    - CUDA GPU (via ``torch.cuda.memory_stats``)

        Logs memory statistics from PyTorch caching allocator (all in bytes).
        GPU compute utilization is not logged by default.

        - General Memory Usage:

            - ``allocated_bytes.all.current`` — Current allocated GPU memory
            - ``allocated_bytes.all.peak`` — Peak allocated GPU memory
            - ``reserved_bytes.all.current`` — Current reserved GPU memory (allocated + cached)
            - ``reserved_bytes.all.peak`` — Peak reserved GPU memory
            - ``active_bytes.all.current`` — Current GPU memory in active use
            - ``active_bytes.all.peak`` — Peak GPU memory in active use
            - ``inactive_split_bytes.all.current`` — Memory in inactive, splittable blocks

        - Allocator Pool Statistics* (for ``small_pool`` and ``large_pool``):

            - ``allocated_bytes.{pool_type}.current`` / ``allocated_bytes.{pool_type}.peak``
            - ``reserved_bytes.{pool_type}.current`` / ``reserved_bytes.{pool_type}.peak``
            - ``active_bytes.{pool_type}.current`` / ``active_bytes.{pool_type}.peak``

        - Allocator Events:

            - ``num_ooms`` — Cumulative out-of-memory errors
            - ``num_alloc_retries`` — Number of allocation retries
            - ``num_device_alloc`` — Number of device allocations
            - ``num_device_free`` — Number of device deallocations

        For a full list of CUDA memory stats, see the
        `PyTorch documentation <https://docs.pytorch.org/docs/stable//generated/torch.cuda.device_memory_used.html>`_.

    - TPU (via ``torch_xla``)

        - *Memory Metrics* (per device, e.g., ``xla:0``):

            - ``memory.free.xla:0`` — Free HBM memory (MB)
            - ``memory.used.xla:0`` — Used HBM memory (MB)
            - ``memory.percent.xla:0`` — Percentage of HBM memory used (%)

        - *XLA Operation Counters*:

            - ``CachedCompile.xla``
            - ``CreateXlaTensor.xla``
            - ``DeviceDataCacheMiss.xla``
            - ``UncachedCompile.xla``
            - ``xla::add.xla``, ``xla::addmm.xla``, etc.

        These counters can be retrieved using: ``torch_xla.debug.metrics.counter_names()``

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


def _prefix_metric_keys(metrics_dict: dict[str, float], prefix: str, separator: str) -> dict[str, float]:
    return {prefix + separator + k: v for k, v in metrics_dict.items()}
