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
import time
from typing import Any, Callable, Optional

from lightning import Callback, LightningModule, Trainer
from lightning.fabric.utilities.speed_monitor import _get_flops_available, _plugin_to_compute_dtype, _SpeedMonitorBase
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class SpeedMonitorCallback(Callback):
    def __init__(self, length_fn: Callable[[Any], int], batch_size: int, **kwargs: Any) -> None:
        super().__init__()
        self.speed_monitor: Optional[_SpeedMonitorBase] = None
        self.speed_monitor_kwargs = kwargs
        self.length_fn = length_fn
        self.batch_size = batch_size
        self.eval_t0: int = 0
        self.train_t0: int = 0
        self.total_lengths: int = 0

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.speed_monitor is not None:
            return  # already setup
        dtype = _plugin_to_compute_dtype(trainer.precision_plugin)
        flops_available = _get_flops_available(trainer.strategy.root_device, dtype)
        self.speed_monitor = _SpeedMonitorBase(flops_available, trainer.logger.log_metrics, **self.speed_monitor_kwargs)

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.fit_loop._should_accumulate():
            return

        self.train_t0 = time.perf_counter()

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        self.total_lengths += self.length_fn(batch)
        if trainer.fit_loop._should_accumulate():
            return
        train_elapsed = time.perf_counter() - self.train_t0
        assert self.speed_monitor is not None
        iter_num = trainer.fit_loop.total_batch_idx
        assert (measured_flops := pl_module.measured_flops) is not None
        self.speed_monitor.on_train_batch_end(
            (iter_num + 1) * self.batch_size,
            train_elapsed,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            trainer.world_size,
            flops_per_batch=measured_flops,
            lengths=self.total_lengths,
        )

    @rank_zero_only
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.eval_t0 = time.perf_counter()

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        eval_elapsed = time.perf_counter() - self.eval_t0
        assert self.speed_monitor is not None
        self.speed_monitor.eval_end(eval_elapsed)
