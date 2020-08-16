"""

GPU Usage Logger
====================

Log GPU memory and GPU usage during training

"""

import os
import shutil
import subprocess
import time

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class GpuUsageLogger(Callback):
    r"""
    Automatically logs GPU memory and GPU usage during training stage.

    Args:
        memory_utilisation: Set to ``True`` to log used, free and percentage of memory
            utilisation at starts and ends of each step. Default: ``True``.
            From nvidia-smi --help-query-gpu
            memory.used = ```Total memory allocated by active contexts.```
            memory.free = ```Total free memory.```
        gpu_utilisation: Set to ``True`` to log percentage of GPU utilisation.
            at starts and ends of each step. Default: ``True``.
        intra_step_time: Set to ``True`` to log the time of each step. Default: ``False``
        inter_step_time: Set to ``True`` to log the time between the end of one step
            and the start of the next. Default: ``False``
        fan_speed: Set to ``True`` to log percentage of fan speed. Default: ``False``.
        temperature: Set to ``True`` to log the memory and gpu temperature in degrees C.
            Default: ``False``

    Example::

        >> from pytorch_lightning import Trainer
        >> from pytorch_lightning.callbacks import GpuUsageLogger
        >> gpu_usage = GpuUsageLogger()
        >> trainer = Trainer(callbacks=[gpu_usage])

    Gpu usage is mainly based on nvidia-smi --query-gpu command.
    The description of the queries used here as appears in
    in ``nvidia-smi --help-query-gpu``:

        "fan.speed"
        ```The fan speed value is the percent of maximum speed that the device's fan is currently
        intended to run at. It ranges from 0 to 100 %. Note: The reported speed is the intended
        fan speed. If the fan is physically blocked and unable to spin, this output will not match
        the actual fan speed. Many parts do not report fan speeds because they rely on cooling via
        fans in the surrounding enclosure.```
        "memory.used"
        ```Total memory allocated by active contexts.```
        "memory.free"
        ```Total free memory.```
        "utilization.gpu"
        ```Percent of time over the past sample period during which one or more kernels was executing
        on the GPU. The sample period may be between 1 second and 1/6 second depending on the product.```
        "utilization.memory"
        ```Percent of time over the past sample period during which global (device) memory was being
        read or written. The sample period may be between 1 second and 1/6 second depending on the
        product.```
        "temperature.gpu"
        ```Core GPU temperature. in degrees C.```
        "temperature.memory"
        ```HBM memory temperature. in degrees C.```

    """

    def __init__(self, memory_utilisation: bool = True, gpu_utilisation: bool = True,
                 intra_step_time: bool = False, inter_step_time: bool = False,
                 fan_speed: bool = False, temperature: bool = False):
        super().__init__()

        if shutil.which('nvidia-smi') is None:
            raise MisconfigurationException(
                'Cannot use GpuUsageLogger callback because nvidia driver is not installed.'
            )

        self.memory_utilisation = memory_utilisation
        self.gpu_utilisation = gpu_utilisation
        self.intra_step_time = intra_step_time
        self.inter_step_time = inter_step_time
        self.fan_speed = fan_speed
        self.temperature = temperature
        self.snap_intra_step_time = None
        self.snap_inter_step_time = None

    def on_batch_start(self, trainer, pl_module):
        if self.gpu_utilisation:
            self._log_gpu(trainer)
        if self.memory_utilisation:
            self._log_memory(trainer)

        if self.inter_step_time:
            # First log at beginning of second step
            if self.snap_inter_step_time:
                trainer.logger.log_metrics(
                    {'Batch_Time/inter_step (ms)': (time.time() - self.snap_inter_step_time) * 1000},
                    step=trainer.global_step
                )

        if self.intra_step_time:
            self.snap_intra_step_time = time.time()

    def on_batch_end(self, trainer, pl_module):
        if self.gpu_utilisation:
            self._log_gpu(trainer)
        if self.memory_utilisation:
            self._log_memory(trainer)

        if self.fan_speed:
            trainer.logger.log_metrics(self._get_gpu_stat("fan.speed", "%"), step=trainer.global_step)
        if self.temperature:
            trainer.logger.log_metrics(self._get_gpu_stat("temperature.gpu", "degrees C"), step=trainer.global_step)
            trainer.logger.log_metrics(self._get_gpu_stat("temperature.memory", "degrees C"), step=trainer.global_step)

        if self.inter_step_time:
            self.snap_inter_step_time = time.time()

        if self.intra_step_time:
            if self.snap_intra_step_time:
                trainer.logger.log_metrics(
                    {'Batch_Time/intra_step (ms)': (time.time() - self.snap_intra_step_time) * 1000},
                    step=trainer.global_step
                )

    def on_train_start(self, trainer, pl_module):
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use GpuUsageLogger callback with Trainer that has no logger.'
            )

    def on_train_epoch_start(self, trainer, pl_module):
        self.snap_intra_step_time = None
        self.snap_inter_step_time = None

    @staticmethod
    def _get_gpu_stat(pitem: str, unit: str):
        result = subprocess.run(
            [shutil.which('nvidia-smi'), f"--query-gpu={pitem}", "--format=csv,nounits,noheader"],
            encoding="utf-8", stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
            check=True
        )

        try:
            gpu_usage = [float(x) for x in result.stdout.strip().split(os.linesep)]
        except ValueError:
            gpu_usage = [0]

        return {f"GPU_{pitem}/gpu_id_{index} ({unit})": usage for index, usage in enumerate(gpu_usage)}

    def _log_gpu(self, trainer):
        trainer.logger.log_metrics(self._get_gpu_stat("utilization.gpu", "%"), step=trainer.global_step)

    def _log_memory(self, trainer):
        trainer.logger.log_metrics(self._get_gpu_stat("memory.used", "MB"), step=trainer.global_step)
        trainer.logger.log_metrics(self._get_gpu_stat("memory.free", "MB"), step=trainer.global_step)
        trainer.logger.log_metrics(self._get_gpu_stat("utilization.memory", "%"), step=trainer.global_step)
