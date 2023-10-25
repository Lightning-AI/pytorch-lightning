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
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch

from lightning.fabric.plugins import Precision
from lightning.fabric.utilities.throughput_monitor import (
    _get_flops_available,
    _ThroughputMonitorBase,
)
from lightning.fabric.utilities.throughput_monitor import (
    _plugin_to_compute_dtype as fabric_plugin_to_compute_dtype,
)
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.plugins import (
    DoublePrecisionPlugin,
    FSDPPrecisionPlugin,
    MixedPrecisionPlugin,
    PrecisionPlugin,
    TransformerEnginePrecisionPlugin,
)
from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecisionPlugin
from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
from lightning.pytorch.plugins.precision.half import HalfPrecisionPlugin
from lightning.pytorch.plugins.precision.xla import XLAPrecisionPlugin
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class ThroughputMonitor(Callback):
    """Tracks and logs throughput.

    +--------------------------+---------------------------------------------------------------------------------------+
    | Key                      | Value                                                                                 |
    +==========================+=======================================================================================+
    | `batches_per_sec`        | Rolling average (over `window_size` most recent batches) of the number of batches     |
    |                          | processed per second                                                                  |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `samples_per_sec`        | Rolling average (over `window_size` most recent batches) of the number of samples     |
    |                          | processed per second                                                                  |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `items_per_sec`          | Rolling average (over `window_size` most recent batches) of the number of items       |
    |                          | processed per second                                                                  |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `flops_per_sec`          | Estimates flops by `flops_per_batch * batches_per_sec`                                |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `device/batches_per_sec` | `batches_per_sec` divided by world size                                               |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `device/samples_per_sec` | `samples_per_sec` divided by world size                                               |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `device/items_per_sec`   | `items_per_sec` divided by world size. This may include padding depending on the data |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `device/flops_per_sec`   | `flops_per_sec` divided by world size. Only logged when model has attribute           |
    |                          | `flops_per_batch`                                                                     |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `device/mfu`             | `device/flops_per_sec` divided by world size.                                         |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `time/train`             | Total elapsed training time                                                           |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `time/val`               | Total elapsed validation time                                                         |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `time/total`             | Total elapsed time (time/train + time/val)                                            |
    +--------------------------+---------------------------------------------------------------------------------------+

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
        self._monitor: Optional[_ThroughputMonitorBase] = None
        self._kwargs = kwargs
        self.length_fn = length_fn
        self.batch_size_fn = batch_size_fn
        self._eval_t0 = 0.0
        self._train_t0 = 0.0
        self._total_lengths = 0

    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        if self._monitor is not None:
            return  # already setup
        dtype = _plugin_to_compute_dtype(trainer.precision_plugin)
        flops_available = _get_flops_available(trainer.strategy.root_device, dtype)
        if flops_available is not None and not hasattr(pl_module, "flops_per_batch"):
            rank_zero_info(
                "When using the `ThroughputMonitor`, you need to define a `flops_per_batch` attribute or property"
                f" in {pl_module} to compute the FLOPs."
            )
        self._monitor = _ThroughputMonitorBase(flops_available, **self._kwargs)

    @rank_zero_only
    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self._train_t0 = time.perf_counter()

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        train_elapsed = time.perf_counter() - self._train_t0
        self._total_lengths += self.length_fn(batch)
        if trainer.fit_loop._should_accumulate():
            # returning here assumes that `flops_per_batch` will include the backward flops
            return
        flops_per_batch = pl_module.flops_per_batch if hasattr(pl_module, "flops_per_batch") else None
        assert self._monitor is not None
        iter_num = trainer.fit_loop.total_batch_idx
        batch_size = self.batch_size_fn(batch)
        samples = (iter_num + 1) * batch_size
        metrics = self._monitor.on_train_batch_end(
            samples,
            train_elapsed,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            trainer.world_size,
            flops_per_batch=flops_per_batch,
            lengths=self._total_lengths,
        )
        trainer._logger_connector.log_metrics(metrics)

    @rank_zero_only
    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if trainer.sanity_checking:
            return
        self._eval_t0 = time.perf_counter()

    @rank_zero_only
    def on_validation_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if trainer.sanity_checking:
            return
        eval_elapsed = time.perf_counter() - self._eval_t0
        assert self._monitor is not None
        self._monitor.eval_end(eval_elapsed)


def _plugin_to_compute_dtype(plugin: Union[Precision, PrecisionPlugin]) -> torch.dtype:
    if not isinstance(plugin, PrecisionPlugin):
        return fabric_plugin_to_compute_dtype(plugin)
    if isinstance(plugin, BitsandbytesPrecisionPlugin):
        return plugin.dtype
    if isinstance(plugin, HalfPrecisionPlugin):
        return plugin._desired_input_dtype
    if isinstance(plugin, MixedPrecisionPlugin):
        return torch.bfloat16 if plugin.precision == "bf16-mixed" else torch.half
    if isinstance(plugin, DoublePrecisionPlugin):
        return torch.double
    if isinstance(plugin, (XLAPrecisionPlugin, DeepSpeedPrecisionPlugin)):
        return plugin._desired_dtype
    if isinstance(plugin, TransformerEnginePrecisionPlugin):
        return torch.int8
    if isinstance(plugin, FSDPPrecisionPlugin):
        return plugin.mixed_precision_config.reduce_dtype or torch.float32
    if isinstance(plugin, PrecisionPlugin):
        return torch.float32
    raise NotImplementedError(plugin)
