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
GPU Stats Monitor
=================

Monitor and logs GPU stats during training.

"""

import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


class GPUStatsMonitor(Callback):
    r"""
    .. deprecated:: v1.5
        The `GPUStatsMonitor` callback was deprecated in v1.5 and will be removed in v1.7.
        Please use the `DeviceStatsMonitor` callback instead.

    Automatically monitors and logs GPU stats during training stage. ``GPUStatsMonitor``
    is a callback and in order to use it you need to assign a logger in the ``Trainer``.

    Args:
        memory_utilization: Set to ``True`` to monitor used, free and percentage of memory
            utilization at the start and end of each step. Default: ``True``.
        gpu_utilization: Set to ``True`` to monitor percentage of GPU utilization
            at the start and end of each step. Default: ``True``.
        intra_step_time: Set to ``True`` to monitor the time of each step. Default: ``False``.
        inter_step_time: Set to ``True`` to monitor the time between the end of one step
            and the start of the next step. Default: ``False``.
        fan_speed: Set to ``True`` to monitor percentage of fan speed. Default: ``False``.
        temperature: Set to ``True`` to monitor the memory and gpu temperature in degree Celsius.
            Default: ``False``.

    Raises:
        MisconfigurationException:
            If NVIDIA driver is not installed, not running on GPUs, or ``Trainer`` has no logger.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import GPUStatsMonitor
        >>> gpu_stats = GPUStatsMonitor() # doctest: +SKIP
        >>> trainer = Trainer(callbacks=[gpu_stats]) # doctest: +SKIP

    GPU stats are mainly based on `nvidia-smi --query-gpu` command. The description of the queries is as follows:

    - **fan.speed** – The fan speed value is the percent of maximum speed that the device's fan is currently
      intended to run at. It ranges from 0 to 100 %. Note: The reported speed is the intended fan speed.
      If the fan is physically blocked and unable to spin, this output will not match the actual fan speed.
      Many parts do not report fan speeds because they rely on cooling via fans in the surrounding enclosure.
    - **memory.used** – Total memory allocated by active contexts.
    - **memory.free** – Total free memory.
    - **utilization.gpu** – Percent of time over the past sample period during which one or more kernels was
      executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product.
    - **utilization.memory** – Percent of time over the past sample period during which global (device) memory was
      being read or written. The sample period may be between 1 second and 1/6 second depending on the product.
    - **temperature.gpu** – Core GPU temperature, in degrees C.
    - **temperature.memory** – HBM memory temperature, in degrees C.

    """

    def __init__(
        self,
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        intra_step_time: bool = False,
        inter_step_time: bool = False,
        fan_speed: bool = False,
        temperature: bool = False,
    ):
        super().__init__()

        rank_zero_deprecation(
            "The `GPUStatsMonitor` callback was deprecated in v1.5 and will be removed in v1.7."
            " Please use the `DeviceStatsMonitor` callback instead."
        )

        if shutil.which("nvidia-smi") is None:
            raise MisconfigurationException(
                "Cannot use GPUStatsMonitor callback because NVIDIA driver is not installed."
            )

        self._log_stats = AttributeDict(
            {
                "memory_utilization": memory_utilization,
                "gpu_utilization": gpu_utilization,
                "intra_step_time": intra_step_time,
                "inter_step_time": inter_step_time,
                "fan_speed": fan_speed,
                "temperature": temperature,
            }
        )

        # The logical device IDs for selected devices
        self._device_ids: List[int] = []  # will be assigned later in setup()

        # The unmasked real GPU IDs
        self._gpu_ids: List[str] = []  # will be assigned later in setup()

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if not trainer.loggers:
            raise MisconfigurationException("Cannot use GPUStatsMonitor callback with Trainer that has no logger.")

        if trainer.strategy.root_device.type != "cuda":
            raise MisconfigurationException(
                "You are using GPUStatsMonitor but are not running on GPU."
                f" The root device type is {trainer.strategy.root_device.type}."
            )

        # The logical device IDs for selected devices
        self._device_ids = sorted(set(trainer.device_ids))

        # The unmasked real GPU IDs
        self._gpu_ids = self._get_gpu_ids(self._device_ids)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._snap_intra_step_time: Optional[float] = None
        self._snap_inter_step_time: Optional[float] = None

    @rank_zero_only
    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if self._log_stats.intra_step_time:
            self._snap_intra_step_time = time.time()

        if not trainer._logger_connector.should_update_logs:
            return

        gpu_stat_keys = self._get_gpu_stat_keys()
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(self._device_ids, gpu_stats, gpu_stat_keys)

        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs["batch_time/inter_step (ms)"] = (time.time() - self._snap_inter_step_time) * 1000

        for logger in trainer.loggers:
            logger.log_metrics(logs, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if not trainer._logger_connector.should_update_logs:
            return

        gpu_stat_keys = self._get_gpu_stat_keys() + self._get_gpu_device_stat_keys()
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(self._device_ids, gpu_stats, gpu_stat_keys)

        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            logs["batch_time/intra_step (ms)"] = (time.time() - self._snap_intra_step_time) * 1000

        for logger in trainer.loggers:
            logger.log_metrics(logs, step=trainer.fit_loop.epoch_loop._batches_that_stepped)

    @staticmethod
    def _get_gpu_ids(device_ids: List[int]) -> List[str]:
        """Get the unmasked real GPU IDs."""
        # All devices if `CUDA_VISIBLE_DEVICES` unset
        default = ",".join(str(i) for i in range(torch.cuda.device_count()))
        cuda_visible_devices: List[str] = os.getenv("CUDA_VISIBLE_DEVICES", default=default).split(",")
        return [cuda_visible_devices[device_id].strip() for device_id in device_ids]

    def _get_gpu_stats(self, queries: List[str]) -> List[List[float]]:
        if not queries:
            return []

        """Run nvidia-smi to get the gpu stats"""
        gpu_query = ",".join(queries)
        format = "csv,nounits,noheader"
        gpu_ids = ",".join(self._gpu_ids)
        result = subprocess.run(
            [
                # it's ok to suppress the warning here since we ensure nvidia-smi exists during init
                shutil.which("nvidia-smi"),  # type: ignore
                f"--query-gpu={gpu_query}",
                f"--format={format}",
                f"--id={gpu_ids}",
            ],
            encoding="utf-8",
            capture_output=True,
            check=True,
        )

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except ValueError:
                return 0.0

        stats = [[_to_float(x) for x in s.split(", ")] for s in result.stdout.strip().split(os.linesep)]
        return stats

    @staticmethod
    def _parse_gpu_stats(
        device_ids: List[int], stats: List[List[float]], keys: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Parse the gpu stats into a loggable dict."""
        logs = {}
        for i, device_id in enumerate(device_ids):
            for j, (x, unit) in enumerate(keys):
                logs[f"device_id: {device_id}/{x} ({unit})"] = stats[i][j]
        return logs

    def _get_gpu_stat_keys(self) -> List[Tuple[str, str]]:
        """Get the GPU stats keys."""
        stat_keys = []

        if self._log_stats.gpu_utilization:
            stat_keys.append(("utilization.gpu", "%"))

        if self._log_stats.memory_utilization:
            stat_keys.extend([("memory.used", "MB"), ("memory.free", "MB"), ("utilization.memory", "%")])

        return stat_keys

    def _get_gpu_device_stat_keys(self) -> List[Tuple[str, str]]:
        """Get the device stats keys."""
        stat_keys = []

        if self._log_stats.fan_speed:
            stat_keys.append(("fan.speed", "%"))

        if self._log_stats.temperature:
            stat_keys.extend([("temperature.gpu", "°C"), ("temperature.memory", "°C")])

        return stat_keys
