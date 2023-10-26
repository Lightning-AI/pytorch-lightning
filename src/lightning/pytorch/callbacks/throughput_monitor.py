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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import torch
from lightning_utilities.core.rank_zero import rank_zero_warn

from lightning.fabric.plugins import Precision
from lightning.fabric.utilities.throughput import Throughput, get_available_flops
from lightning.fabric.utilities.throughput import _plugin_to_compute_dtype as fabric_plugin_to_compute_dtype
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
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities.rank_zero import rank_zero_only

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class ThroughputMonitor(Callback):
    r"""Computes and logs throughput with the :class:`lightning.fabric.utilities.throughput.Throughput`

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
        self.available_flops: Optional[int] = None
        self._throughputs: Dict[RunningStage, Throughput] = {}
        self._t0s: Dict[RunningStage, float] = {}
        self._lengths: Dict[RunningStage, int] = {}

    def setup(self, trainer: "Trainer", pl_module: "LightningModule", stage: TrainerFn) -> None:
        dtype = _plugin_to_compute_dtype(trainer.precision_plugin)
        self.available_flops = get_available_flops(trainer.strategy.root_device, dtype)

        if stage == TrainerFn.FITTING and trainer.enable_validation:
            # `fit` includes validation inside
            throughput = Throughput(available_flops=self.available_flops, world_size=trainer.world_size, **self.kwargs)
            self._throughputs[RunningStage.VALIDATING] = throughput

        throughput = Throughput(available_flops=self.available_flops, world_size=trainer.world_size, **self.kwargs)
        stage = trainer.state.stage
        assert stage is not None
        self._throughputs[stage] = throughput

    def _start(self, trainer: "Trainer") -> None:
        stage = trainer.state.stage
        assert stage is not None
        self._throughputs[stage].reset()
        self._lengths[stage] = 0
        self._t0s[stage] = time.perf_counter()

    @torch.inference_mode()  # in case `length_fn` or `batch_size_fn` computes grads
    def _update(self, trainer: "Trainer", pl_module: "LightningModule", batch: Any, iter_num: int) -> None:
        stage = trainer.state.stage
        assert stage is not None
        throughput = self._throughputs[stage]

        if trainer.strategy.root_device.type == "cuda":
            # required or else perf_counter() won't be correct
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._t0s[stage]
        if self.length_fn is not None:
            self._lengths[stage] += self.length_fn(batch)

        if hasattr(pl_module, "flops_per_batch"):
            flops_per_batch = pl_module.flops_per_batch
        else:
            rank_zero_warn(
                "When using the `ThroughputMonitor`, you need to define a `flops_per_batch` attribute or property"
                f" in {type(pl_module).__name__} to compute the FLOPs."
            )
            flops_per_batch = None

        batch_size = self.batch_size_fn(batch)
        throughput.update(
            time=elapsed,
            # this assumes that all iterations used the same batch size
            samples=iter_num * batch_size,
            flops_per_batch=flops_per_batch,
            lengths=None if self.length_fn is None else self._lengths[stage],
        )

    def _compute(self, trainer: "Trainer", step: Optional[int] = None) -> None:
        if not trainer._logger_connector.should_update_logs:
            return
        stage = trainer.state.stage
        assert stage is not None
        throughput = self._throughputs[stage]
        metrics = throughput.compute()
        # prefix with the stage to avoid collisions
        metrics = {f"{stage.value}{throughput.separator}{k}": v for k, v in metrics.items()}
        trainer._logger_connector.log_metrics(metrics, step=step)

    @rank_zero_only
    def on_train_start(self, trainer: "Trainer", *_) -> None:
        self._start(trainer)

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, *_
    ) -> None:
        self._update(trainer, pl_module, batch, trainer.fit_loop.total_batch_idx + 1)
        # log when gradient accumulation is over
        if not trainer.fit_loop._should_accumulate():
            self._compute(trainer)

    @rank_zero_only
    def on_validation_start(self, trainer: "Trainer", *_: Any) -> None:
        if trainer.sanity_checking:
            return
        self._start(trainer)

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, *_: Any, **__: Any
    ) -> None:
        if trainer.sanity_checking:
            return
        self._update(trainer, pl_module, batch, trainer._evaluation_loop.batch_progress.total.ready)
        self._compute(trainer, trainer._evaluation_loop.batch_progress.current.ready)

    def on_validation_end(self, trainer: "Trainer", *_: Any) -> None:
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return
        # add the validation time to the training time before continuing to avoid sinking the training throughput
        time_between_train_and_val = (
            self._t0s[RunningStage.VALIDATING] - self._throughputs[RunningStage.TRAINING]._time[-1]
        )
        val_time = self._throughputs[RunningStage.VALIDATING]._time[-1]
        self._t0s[RunningStage.TRAINING] += time_between_train_and_val + val_time

    @rank_zero_only
    def on_test_start(self, trainer: "Trainer", *_: Any) -> None:
        self._start(trainer)

    @rank_zero_only
    def on_test_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, *_: Any, **__: Any
    ) -> None:
        self._update(trainer, pl_module, batch, trainer._evaluation_loop.batch_progress.total.ready)
        self._compute(trainer, trainer._evaluation_loop.batch_progress.current.ready)

    @rank_zero_only
    def on_predict_start(self, trainer: "Trainer", *_: Any) -> None:
        self._start(trainer)

    @rank_zero_only
    def on_predict_batch_end(
        self, trainer: "Trainer", pl_module: "LightningModule", outputs: Any, batch: Any, *_: Any, **__: Any
    ) -> None:
        self._update(trainer, pl_module, batch, trainer.predict_loop.batch_progress.total.ready)
        self._compute(trainer, trainer.predict_loop.batch_progress.current.ready)


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
