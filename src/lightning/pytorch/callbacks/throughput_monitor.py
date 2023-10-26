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
from lightning.fabric.utilities.throughput import Throughput, _get_flops_available
from lightning.fabric.utilities.throughput import (
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
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class ThroughputMonitor(Callback):
    r"""Tracks and logs throughput with the :class:`lightning.fabric.utilities.throughput.Throughput`

    # FIXME example

    Notes:
        - Only support for :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit` is currently implemented.
        - It assumes that the batch size is the same during all iterations.
        - It will try to access a ``flops_per_batch`` attribute on your ``LightningModule`` on every iteration.
            We suggest using the :func:`lightning.fabric.utilities.throughput.measure_flops` function for this.
            You might want to compute it differently each time based on your setup.

    Args:
        batch_size_fn: A function to compute the number of samples given a batch.
        length_fn: A function to compute the number of items in a sample given a batch.
        \**kwargs: See available parameters in
            :class:`lightning.fabric.utilities.throughput.Throughput`

    """

    def __init__(
        self, batch_size_fn: Callable[[Any], int], length_fn: Optional[Callable[[Any], int]] = None, **kwargs: Any
    ) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.batch_size_fn = batch_size_fn
        self.length_fn = length_fn
        self._throughput: Optional[Throughput] = None
        self._train_t0 = 0.0
        self._total_lengths = 0

    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: str) -> None:
        if stage is not TrainerFn.FITTING:
            # TODO: this could use a monitor per stage
            raise NotImplementedError(f"`trainer.{stage}()` is not supported")

        if self._throughput is not None:
            return  # already setup
        dtype = _plugin_to_compute_dtype(trainer.precision_plugin)
        flops_available = _get_flops_available(trainer.strategy.root_device, dtype)
        if flops_available is not None and not hasattr(pl_module, "flops_per_batch"):
            rank_zero_info(
                "When using the `ThroughputMonitor`, you need to define a `flops_per_batch` attribute or property"
                f" in {pl_module} to compute the FLOPs."
            )
        self._throughput = Throughput(flops_available=flops_available, world_size=trainer.world_size, **self.kwargs)

    @rank_zero_only
    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self._train_t0 = time.perf_counter()

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        train_elapsed = time.perf_counter() - self._train_t0
        if self.length_fn is not None:
            self._total_lengths += self.length_fn(batch)
        if trainer.fit_loop._should_accumulate():
            # FIXME: double check this after `update` is implemented
            # returning here assumes that `flops_per_batch` will include the backward flops
            return
        flops_per_batch = pl_module.flops_per_batch if hasattr(pl_module, "flops_per_batch") else None
        batch_size = self.batch_size_fn(batch)
        iter_num = trainer.fit_loop.total_batch_idx + 1
        assert self._throughput is not None
        metrics = self._throughput.compute(
            time=train_elapsed,
            # this assumes that all iterations used the same batch size
            samples=iter_num * batch_size,
            flops_per_batch=flops_per_batch,
            lengths=None if self.length_fn is None else self._total_lengths,
        )
        # prefix with the stage to avoid collisions
        metrics = {f"{trainer.state.stage.value}{self._throughput.separator}{k}": v for k, v in metrics.items()}
        trainer._logger_connector.log_metrics(metrics)


def _plugin_to_compute_dtype(plugin: Union[Precision, PrecisionPlugin]) -> torch.dtype:
    # TODO: integrate this into the precision plugins
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
