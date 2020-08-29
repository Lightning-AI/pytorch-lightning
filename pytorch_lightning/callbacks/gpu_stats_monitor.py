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

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
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

    def on_train_start(self, trainer, pl_module):
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use GPUStatsMonitor callback with Trainer that has no logger.'
            )

        if not trainer.on_gpu:
            rank_zero_warn(
                'You are using GPUStatsMonitor but are not running on GPU.'
                ' Logged utilization will be independent from your model.', RuntimeWarning
            )

        self._gpu_ids = ','.join(map(str, trainer.data_parallel_device_ids))

    def on_train_epoch_start(self, trainer, pl_module):
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        gpu_stat_keys = []
        gpu_stat_keys.extend(self._get_gpu_stat_keys())

        gpu_stats = self._get_gpu_stats(gpu_stat_keys)

        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            gpu_stats['batch_time/inter_step2 (ms)'] = (time.time() - self._snap_inter_step_time) * 1000

        trainer.logger.log_metrics(gpu_stats, step=trainer.global_step)

        if self._log_stats.intra_step_time:
            self._snap_intra_step_time = time.time()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        gpu_stat_keys = []
        gpu_stat_keys.extend(self._get_gpu_stat_keys())
        gpu_stat_keys.extend(self._get_gpu_device_stat_keys())

        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        gpu_stats = self._get_gpu_stats(gpu_stat_keys)

        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            gpu_stats['batch_time/intra_step2 (ms)'] = (time.time() - self._snap_intra_step_time) * 1000

        trainer.logger.log_metrics(gpu_stats, step=trainer.global_step)

    def _get_gpu_stats(self, gpu_stat_keys):
        gpu_query = ','.join([m[0] for m in gpu_stat_keys])
        format = 'csv,nounits,noheader'

        result = subprocess.run(
            [shutil.which('nvidia-smi'), f'--query-gpu={gpu_query}', f'--format={format}', f'-i={self._gpu_ids}'],
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True
        )

        def _to_float(x):
            try:
                return float(x)
            except ValueError:
                return 0.

        stats = result.stdout.strip().split(os.linesep)
        stats = [list(map(_to_float, x.split(', '))) for x in stats]

        logs = {}
        for i, gpu_id in enumerate(self._gpu_ids.split(',')):
            gpu_stat_keys = [f'gpu_id: {gpu_id}/{x} ({unit})' for x, unit in gpu_stat_keys]
            logs.update(dict(zip(gpu_stat_keys, stats[i])))

        return logs

    def _get_gpu_stat_keys(self):
        stat_keys = []

        if self._log_stats.gpu_utilization:
            stat_keys.append(('utilization.gpu', '%'))

        if self._log_stats.memory_utilization:
            stat_keys.extend([('memory.used', 'MB'), ('memory.free', 'MB'), ('utilization.memory', '%')])

        return stat_keys

    def _get_gpu_device_stat_keys(self):
        stat_keys = []

        if self._log_stats.fan_speed:
            stat_keys.append(('fan.speed', '%'))

        if self._log_stats.temperature:
            stat_keys.extend([('temperature.gpu', '°C'), ('temperature.memory', '°C')])

        return stat_keys
