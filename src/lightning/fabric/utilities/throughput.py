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
# Adapted from https://github.com/mosaicml/composer/blob/f2a2dc820/composer/callbacks/speed_monitor.py
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, TypeVar, Union

import torch
from typing_extensions import Self

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn

if TYPE_CHECKING:
    from lightning.fabric import Fabric
    from lightning.fabric.plugins import Precision

_THROUGHPUT_METRICS = Dict[str, Union[int, float]]


# The API design of this class follows `torchmetrics.Metric` but it doesn't need to be an actual Metric because there's
# no need for synchronization or reduction as it doesn't use Tensors at all.
class Throughput:
    """Computes throughput.

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
    | `device/flops_per_sec`   | `flops_per_sec` divided by world size.                                                |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `device/mfu`             | `device/flops_per_sec` divided by world size.                                         |
    +--------------------------+---------------------------------------------------------------------------------------+
    | `time`                   | Total elapsed time                                                                    |
    +--------------------------+---------------------------------------------------------------------------------------+

    Example::

        # FIXME

    Notes:
        - The implementation assumes that devices FLOPs are all the same as it normalizes by the world size and only
            takes a single ``available_flops`` value.
        - items_per_sec, flops_per_sec and MFU do not account for padding if present. We suggest using
            samples_per_sec or batches_per_sec to measure throughput under this circumstance.

    Args:
        available_flops: Number of theoretical flops available for a single device.
        world_size: Number of devices available across hosts. Global metrics are not included if the world size is 1.
        window_size: Number of batches to use for a rolling average.
        separator: Key separator to use when creating per-device and global metrics.

    """

    def __init__(
        self, available_flops: Optional[float] = None, world_size: int = 1, window_size: int = 100, separator: str = "/"
    ) -> None:
        self.available_flops = available_flops
        self.separator = separator
        assert world_size > 0
        self.world_size = world_size

        # throughput is computed over a window of values. at least 2 is enforced since it looks at the difference
        # between the first and last elements
        assert window_size > 1
        # custom class instead of `deque(maxlen=)` because it's easy for users to mess up their timer/counters and log
        # values that do not increase monotonically. this class will raise an error if that happens.
        self._samples: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._time: _MonotonicWindow[float] = _MonotonicWindow(maxlen=window_size)
        self._lengths: _MonotonicWindow[int] = _MonotonicWindow(maxlen=window_size)
        self._flops: Deque[int] = deque(maxlen=window_size)

    def update(
        self, *, time: float, samples: int, lengths: Optional[int] = None, flops_per_batch: Optional[int] = None
    ) -> Self:
        """Update throughput metrics.

        Args:
            time: Total elapsed time in seconds. Monotonically increasing.
            samples: Total samples seen per device. Monotonically increasing.
            lengths: Total length of the samples seen. Monotonically increasing.
            flops_per_batch: Flops per batch per device. You can easily compute this by using :func:`measure_flops`.
                The value might be different in each device if the batch size is not the same.
                It should be also be different in iterations that accumulate gradients versus iterations that
                ``backward``.

        """
        self._time.append(time)
        self._samples.append(samples)
        if lengths is not None:
            self._lengths.append(lengths)
            if len(self._samples) != len(self._lengths):
                raise RuntimeError(
                    f"If lengths are passed ({len(self._lengths)}), there needs to be the same number of samples"
                    f" ({len(self._samples)})"
                )
        if flops_per_batch is not None:
            # sum of flops per batch across ranks
            self._flops.append(flops_per_batch * self.world_size)
        return self

    def compute(self) -> _THROUGHPUT_METRICS:
        """Compute throughput metrics."""
        metrics = {"time": self._time[-1], "samples": self._samples[-1]}
        add_global_metrics = self.world_size > 1
        # a different but valid design choice would be to still compute all these metrics even if the window of values
        # has not been filled
        if len(self._time) == self._time.maxlen:
            elapsed_batches = len(self._samples) - 1
            elapsed_samples = self._samples[-1] - self._samples[0]
            elapsed_time = self._time[-1] - self._time[0]
            # we are safe from ZeroDivisionError thanks to `_MonotonicWindow`
            dev_samples_per_sec = elapsed_samples / elapsed_time
            dev_batches_per_sec = elapsed_batches / elapsed_time
            metrics.update(
                {
                    f"device{self.separator}batches_per_sec": elapsed_batches / elapsed_time,
                    f"device{self.separator}samples_per_sec": dev_samples_per_sec,
                }
            )
            if add_global_metrics:
                samples_per_sec = dev_batches_per_sec * self.world_size
                metrics.update(
                    {"batches_per_sec": samples_per_sec, "samples_per_sec": dev_samples_per_sec * self.world_size}
                )

            if len(self._lengths) == self._lengths.maxlen:
                elapsed_lengths = int(self._lengths[-1]) - int(self._lengths[0])
                avg_length = elapsed_lengths / elapsed_batches
                if add_global_metrics:
                    metrics["items_per_sec"] = samples_per_sec * avg_length
                metrics[f"device{self.separator}items_per_sec"] = dev_samples_per_sec * avg_length

        if len(self._flops) == self._flops.maxlen:
            elapsed_flops = sum(self._flops) - self._flops[0]
            elapsed_time = self._time[-1] - self._time[0]
            flops_per_sec = elapsed_flops / elapsed_time
            dev_flops_per_sec = flops_per_sec / self.world_size
            if add_global_metrics:
                metrics["flops_per_sec"] = flops_per_sec
            metrics[f"device{self.separator}flops_per_sec"] = dev_flops_per_sec
            if self.available_flops:
                metrics[f"device{self.separator}mfu"] = dev_flops_per_sec / self.available_flops

        return metrics

    def reset(self) -> Self:
        self._samples.clear()
        self._time.clear()
        self._lengths.clear()
        self._flops.clear()
        return self


class ThroughputMonitor(Throughput):
    r"""Computes throughput.

    This class will automatically keep a count of the number of log calls (``step``). But that can be modified as
    desired. For manual logging, using :class:`Throughput` directly might be desired.

    Example::

        # FIXME: improve, add step example
        monitor = ThroughputMonitor(fabric)
        monitor.compute_and_log()

    Args:
        fabric: The Fabric object.
        \**kwargs: See available parameters in :class:`Throughput`

    """

    def __init__(self, fabric: "Fabric", **kwargs: Any) -> None:
        fabric._validate_launched()  # otherwise world_size might be incorrect
        dtype = _plugin_to_compute_dtype(fabric.strategy.precision)
        available_flops = get_available_flops(fabric.device, dtype)
        super().__init__(available_flops=available_flops, world_size=fabric.world_size, **kwargs)
        self._fabric = fabric
        self.step = -1

        self.update = rank_zero_only(self.update, default=self)  # type: ignore[method-assign]
        self.compute = rank_zero_only(self.compute, default={})  # type: ignore[method-assign]
        self.reset = rank_zero_only(self.reset, default=self)  # type: ignore[method-assign]

    @rank_zero_only
    def compute_and_log(self, step: Optional[int] = None, **kwargs: Any) -> _THROUGHPUT_METRICS:
        r"""See :meth:`Throughput.compute`

        Args:
            step: Can be used to override the logging step.
            \**kwargs: See available parameters in :meth:`Throughput.compute`

        """
        self.step = (self.step + 1) if step is None else step
        metrics = self.compute(**kwargs)
        self._fabric.log_dict(metrics=metrics, step=self.step)
        return metrics


def measure_flops(
    model: torch.nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> int:
    """Utility to compute the total number of FLOPs used by a module during training or during inference.

    It's recommended to create a meta-device model for this:

    Example::

        with torch.device("meta"):
            model = MyModel()
            x = torch.randn(2, 32)

        model_fwd = lambda: model(x)
        fwd_flops = measure_flops(model, model_fwd)

        model_loss = lambda y: y.sum()
        fwd_and_bwd_flops = measure_flops(model, model_fwd, model_loss)

    Args:
        model: The model whose FLOPs should be measured.
        forward_fn: A function that runs ``forward`` on the model and returns the result.
        loss_fn: A function that computes the loss given the ``forward_fn`` output. If provided, the loss and `backward`
            FLOPs will be included in the result.

    """
    if not _TORCH_GREATER_EQUAL_2_1:
        raise ImportError("`measure_flops` requires PyTorch >= 2.1.")
    from torch.utils.flop_counter import FlopCounterMode

    flop_counter = FlopCounterMode(model, display=False)
    with flop_counter:
        if loss_fn is None:
            forward_fn()
        else:
            loss_fn(forward_fn()).backward()
    return flop_counter.get_total_flops()


_CUDA_FLOPS = {
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    "h100-sxm": {
        torch.float64: 67e12,
        torch.float32: 67e12,
        torch.bfloat16: 1.979e15 / 2,
        torch.float16: 1.979e15 / 2,
        torch.int8: 3.958e15 / 2,
    },
    "h100-pcie": {
        torch.float64: 51e12,
        torch.float32: 51e12,
        torch.bfloat16: 1.513e15 / 2,
        torch.float16: 1.513e15 / 2,
        torch.int8: 3.026e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {torch.float64: 19.5e12, torch.float32: 19.5e12, torch.bfloat16: 312e12, torch.float16: 312e12},
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    "a10g": {torch.float32: 31.2e12, torch.bfloat16: 125e12, torch.float16: 125e12},
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100-sxm": {torch.float64: 7.8e12, torch.float32: 15.7e12, torch.float16: 125e12},
    "v100-pcie": {torch.float64: 7e12, torch.float32: 14e12, torch.float16: 112e12},
    "v100s-pcie": {torch.float64: 8.2e12, torch.float32: 16.4e12, torch.float16: 130e12},
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    "t4": {torch.float32: 8.1e12, torch.float16: 65e12, torch.int8: 130e12},
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-5000-data-sheet-us-nvidia-704120-r4-web.pdf
    "quadro rtx 5000": {torch.float32: 11.2e12, torch.float16: 89.2e12},
}

_TPU_FLOPS = {
    # flop count for each TPU generation is the same for all precisions
    # since bfloat16 precision is always used for performing matrix operations
    # for more info: https://cloud.google.com/tpu/docs/bfloat16#choosing_bfloat16
    # source: https://arxiv.org/pdf/1907.10701.pdf
    "v2": 45e12,
    # source: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v3
    "v3": 123e12,
    # source: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v4
    "v4": 275e12,
    # source: https://cloud.google.com/tpu/docs/v5e-training
    "v5litepod": 197e12,
}


def get_available_flops(device: torch.device, dtype: torch.dtype) -> Optional[int]:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        chip = device_name.lower()
        if "h100" in chip and "hbm3" in chip:
            chip = "h100-sxm"
        elif "h100" in chip and ("pcie" in chip or "hbm2e" in chip):
            chip = "h100-pcie"
        elif "a100" in chip:
            chip = "a100"
        elif "a10g" in chip:
            chip = "a10g"
        elif "v100-sxm" in chip:
            chip = "v100-sxm"
        elif "v100-pcie" in chip:
            chip = "v100-pcie"
        elif "t4" in chip:
            chip = "t4"
        elif "quadro rtx 5000" in chip:
            chip = "quadro rtx 5000"
        else:
            # the flops list is not exhaustive, return with a warning
            rank_zero_warn(f"FLOPs not found for {device_name!r}")
            return None
        if chip not in _CUDA_FLOPS:
            # if we were able to parse the chip, it should be in the flops list
            raise RuntimeError(f"FLOPs not found for {device_name!r}, chip is {chip!r}")
        dtype_to_flops = _CUDA_FLOPS[chip]
        if dtype not in dtype_to_flops:
            # for example, T4 doesn't support bfloat16. it might also be that we are missing this dtype from the list
            rank_zero_warn(f"{device_name!r} does not support {dtype}")
            return None
        return int(dtype_to_flops[dtype])

    if device.type == "xla":
        from lightning.fabric.accelerators.xla import _XLA_GREATER_EQUAL_2_1

        if _XLA_GREATER_EQUAL_2_1:
            from torch_xla._internal import tpu
        else:
            from torch_xla.experimental import tpu

        device_name = tpu.get_tpu_env()["TYPE"]
        chip = device_name.lower()
        assert isinstance(device_name, str)
        if chip not in _TPU_FLOPS:
            rank_zero_warn(f"FLOPs not found for TPU {device_name!r} with {dtype}")
            return None
        return int(_TPU_FLOPS[chip])


def _plugin_to_compute_dtype(plugin: "Precision") -> torch.dtype:
    # TODO: integrate this into the precision plugins
    from lightning.fabric.plugins import (
        BitsandbytesPrecision,
        DeepSpeedPrecision,
        DoublePrecision,
        FSDPPrecision,
        HalfPrecision,
        MixedPrecision,
        Precision,
        TransformerEnginePrecision,
        XLAPrecision,
    )

    if not isinstance(plugin, Precision):
        raise RuntimeError(f"Expected a precision plugin, got {plugin}")
    if isinstance(plugin, BitsandbytesPrecision):
        return plugin.dtype
    if isinstance(plugin, (HalfPrecision, MixedPrecision)):
        return plugin._desired_input_dtype
    if isinstance(plugin, DoublePrecision):
        return torch.double
    if isinstance(plugin, (XLAPrecision, DeepSpeedPrecision)):
        return plugin._desired_dtype
    if isinstance(plugin, TransformerEnginePrecision):
        return torch.int8
    if isinstance(plugin, FSDPPrecision):
        return plugin.mixed_precision_config.reduce_dtype or torch.float32
    if isinstance(plugin, Precision):
        return torch.float32
    raise NotImplementedError(plugin)


T = TypeVar("T", bound=float)


class _MonotonicWindow(List[T]):
    """Custom fixed size list that only supports right-append and ensures that all values increase monotonically."""

    def __init__(self, maxlen: int) -> None:
        super().__init__()
        self.maxlen = maxlen

    @property
    def last(self) -> Optional[T]:
        if len(self) > 0:
            return self[-1]
        return None

    def append(self, x: T) -> None:
        last = self.last
        if last is not None and last >= x:
            raise ValueError(f"Expected the value to increase, last: {last}, current: {x}")
        list.append(self, x)
        # truncate excess
        if len(self) > self.maxlen:
            del self[0]

    def __setitem__(self, key: Any, value: Any) -> None:
        # assigning is not implemented since we don't use it. it could be by checking all previous values
        raise NotImplementedError("__setitem__ is not supported")
