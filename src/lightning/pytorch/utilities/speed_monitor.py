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
    """Logs throughput and utilization.

    +-------------------------------------+--------------------------------------------------------+
    | Key                                 | Logged data                                            |
    +=====================================+========================================================+
    |                                     | Rolling average (over `window_size` most recent        |
    | `throughput/batches_per_sec`        | batches) of the number of batches processed per second |
    |                                     |                                                        |
    +-------------------------------------+--------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent        |
    | `throughput/samples_per_sec`        | batches) of the number of samples processed per second |
    |                                     |                                                        |
    +-------------------------------------+--------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent        |
    | `throughput/items_per_sec`          | batches) of the number of items processed per second.  |
    |                                     | This may include padding depending on the data         |
    +-------------------------------------+--------------------------------------------------------+
    |                                     | Estimates flops by `flops_per_batch * batches_per_sec` |
    | `throughput/flops_per_sec`          |                                                        |
    |                                     |                                                        |
    +-------------------------------------+--------------------------------------------------------+
    | `throughput/device/batches_per_sec` | `throughput/batches_per_sec` divided by world size     |
    +-------------------------------------+--------------------------------------------------------+
    | `throughput/device/samples_per_sec` | `throughput/samples_per_sec` divided by world size     |
    +-------------------------------------+--------------------------------------------------------+
    |                                     | `throughput/items_per_sec` divided by world size. This |
    | `throughput/device/items_per_sec`   | may include padding depending on the data              |
    |                                     |                                                        |
    +-------------------------------------+--------------------------------------------------------+
    |                                     | `throughput/flops_per_sec` divided by world size. Only |
    | `throughput/device/flops_per_sec`   | logged when model has attribute `flops_per_batch`      |
    |                                     |                                                        |
    +-------------------------------------+--------------------------------------------------------+
    |                                     | `throughput/device/flops_per_sec` divided by world     |
    |                                     |  size.                                                 |
    | `throughput/device/mfu`             |                                                        |
    |                                     |                                                        |
    +-------------------------------------+--------------------------------------------------------+
    | `time/train`                        | Total elapsed training time                            |
    +-------------------------------------+--------------------------------------------------------+
    | `time/val`                          | Total elapsed validation time                          |
    +-------------------------------------+--------------------------------------------------------+
    | `time/total`                        | Total elapsed time (time/train + time/val)             |
    +-------------------------------------+--------------------------------------------------------+

    Notes:
        - The implementation assumes that devices are homogeneous as it normalizes by the world size.
        - items_per_sec, flops_per_sec and MFU do not account for padding if present. We suggest using
            samples_per_sec or batches_per_sec to measure throughput under this circumstance.

    Args:
        length_fn: A function to compute the number of items in a sample given a batch.
        batch_size_fn: A function to compute the number of samples given a batch.
        window_size: Number of batches to use for a rolling average of throughput.
        time_unit: Time unit to use for `time` logging.

    """

    def __init__(self, length_fn: Callable[[Any], int], batch_size_fn: Callable[[Any], int], **kwargs: Any) -> None:
        super().__init__()
        self._speed_monitor: Optional[_SpeedMonitorBase] = None
        self._kwargs = kwargs
        self.length_fn = length_fn
        self.batch_size_fn = batch_size_fn
        self._eval_t0 = 0.0
        self._train_t0 = 0.0
        self._total_lengths = 0

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self._speed_monitor is not None:
            return  # already setup
        dtype = _plugin_to_compute_dtype(trainer.precision_plugin)
        flops_available = _get_flops_available(trainer.strategy.root_device, dtype)
        # FIXME: multiple loggers
        self._speed_monitor = _SpeedMonitorBase(flops_available, trainer.logger.log_metrics, **self._kwargs)

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._train_t0 = time.perf_counter()

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        train_elapsed = time.perf_counter() - self._train_t0
        self._total_lengths += self.length_fn(batch)
        if trainer.fit_loop._should_accumulate():
            return
        # FIXME: check in each call
        flops_per_batch = pl_module.flops_per_batch if hasattr(pl_module, "flops_per_batch") else None
        assert self._speed_monitor is not None
        iter_num = trainer.fit_loop.total_batch_idx
        batch_size = self.batch_size_fn(batch)
        samples = (iter_num + 1) * batch_size
        self._speed_monitor.on_train_batch_end(
            samples,
            train_elapsed,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            trainer.world_size,
            flops_per_batch=flops_per_batch,
            lengths=self._total_lengths,
        )

    @rank_zero_only
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._eval_t0 = time.perf_counter()

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        eval_elapsed = time.perf_counter() - self._eval_t0
        assert self._speed_monitor is not None
        self._speed_monitor.eval_end(eval_elapsed)
