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
from typing import List, Tuple, Dict

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import AttributeDict


class GPUStatsMonitor(Callback):
    r"""
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
        temperature: bool = False
    ):
        super().__init__()

        if shutil.which('nvidia-smi') is None:
            raise MisconfigurationException(
                'Cannot use GPUStatsMonitor callback because NVIDIA driver is not installed.'
            )

        self._log_stats = AttributeDict({
            'memory_utilization': memory_utilization,
            'gpu_utilization': gpu_utilization,
            'intra_step_time': intra_step_time,
            'inter_step_time': inter_step_time,
            'fan_speed': fan_speed,
            'temperature': temperature
        })

    def on_train_start(self, trainer, *args, **kwargs):
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use GPUStatsMonitor callback with Trainer that has no logger.'
            )

        if not trainer.on_gpu:
            raise MisconfigurationException(
                'You are using GPUStatsMonitor but are not running on GPU'
                f' since gpus attribute in Trainer is set to {trainer.gpus}.'
            )

        self._gpu_ids = ','.join(map(str, trainer.data_parallel_device_ids))

    def on_train_epoch_start(self, *args, **kwargs):
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(self, trainer, *args, **kwargs):
        if self._log_stats.intra_step_time:
            self._snap_intra_step_time = time.time()

        if not self._should_log(trainer):
            return

        gpu_stat_keys = self._get_gpu_stat_keys()
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(self._gpu_ids, gpu_stats, gpu_stat_keys)

        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs['batch_time/inter_step (ms)'] = (time.time() - self._snap_inter_step_time) * 1000

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(self, trainer, *args, **kwargs):
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if not self._should_log(trainer):
            return

        gpu_stat_keys = self._get_gpu_stat_keys() + self._get_gpu_device_stat_keys()
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(self._gpu_ids, gpu_stats, gpu_stat_keys)

        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            logs['batch_time/intra_step (ms)'] = (time.time() - self._snap_intra_step_time) * 1000

        trainer.logger.log_metrics(logs, step=trainer.global_step)

    def _get_gpu_stats(self, queries: List[str]) -> List[List[float]]:
        """Run nvidia-smi to get the gpu stats"""
        gpu_query = ','.join(queries)
        format = 'csv,nounits,noheader'
        result = subprocess.run(
            [shutil.which('nvidia-smi'), f'--query-gpu={gpu_query}', f'--format={format}', f'--id={self._gpu_ids}'],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True
        )

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except ValueError:
                return 0.

        stats = result.stdout.strip().split(os.linesep)
        stats = [[_to_float(x) for x in s.split(', ')] for s in stats]
        return stats

    @staticmethod
    def _parse_gpu_stats(gpu_ids: str, stats: List[List[float]], keys: List[Tuple[str, str]]) -> Dict[str, float]:
        """Parse the gpu stats into a loggable dict"""
        logs = {}
        for i, gpu_id in enumerate(gpu_ids.split(',')):
            for j, (x, unit) in enumerate(keys):
                logs[f'gpu_id: {gpu_id}/{x} ({unit})'] = stats[i][j]
        return logs

    def _get_gpu_stat_keys(self) -> List[Tuple[str, str]]:
        """Get the GPU stats keys"""
        stat_keys = []

        if self._log_stats.gpu_utilization:
            stat_keys.append(('utilization.gpu', '%'))

        if self._log_stats.memory_utilization:
            stat_keys.extend([('memory.used', 'MB'), ('memory.free', 'MB'), ('utilization.memory', '%')])

        return stat_keys

    def _get_gpu_device_stat_keys(self) -> List[Tuple[str, str]]:
        """Get the device stats keys"""
        stat_keys = []

        if self._log_stats.fan_speed:
            stat_keys.append(('fan.speed', '%'))

        if self._log_stats.temperature:
            stat_keys.extend([('temperature.gpu', '°C'), ('temperature.memory', '°C')])

        return stat_keys

    @staticmethod
    def _should_log(trainer) -> bool:
        should_log = (
            (trainer.global_step + 1) % trainer.log_every_n_steps == 0
            or trainer.should_stop
        )

        should_log = should_log and not trainer.fast_dev_run
        return should_log
