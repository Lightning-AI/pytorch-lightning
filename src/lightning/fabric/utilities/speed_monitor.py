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
from collections import deque
from contextlib import nullcontext
from typing import Any, Callable, Deque, Dict, Optional

import torch
from torch.utils.flop_counter import FlopCounterMode

from lightning import Fabric
from lightning.fabric.accelerators.xla import _XLA_GREATER_EQUAL_2_1
from lightning.fabric.plugins import (
    BitsandbytesPrecision,
    DoublePrecision,
    FSDPPrecision,
    HalfPrecision,
    MixedPrecision,
    Precision,
    TransformerEnginePrecision,
    XLAPrecision,
)
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.plugins import (
    DoublePrecisionPlugin,
    FSDPPrecisionPlugin,
    HalfPrecisionPlugin,
    MixedPrecisionPlugin,
    XLAPrecisionPlugin,
)

_GPU_AVAILABLE_FLOPS = {
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

_TPU_AVAILABLE_FLOPS = {
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


def _get_flops_available(device: torch.device, dtype: torch.dtype) -> Optional[float]:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device).lower()
        if "h100" in device_name and "hbm3" in device_name:
            device_name = "h100-sxm"
        elif "h100" in device_name and ("pcie" in device_name or "hbm2e" in device_name):
            device_name = "h100-pcie"
        elif "a100" in device_name:
            device_name = "a100"
        elif "a10g" in device_name:
            device_name = "a10g"
        elif "v100-sxm" in device_name:
            device_name = "v100-sxm"
        elif "v100-pcie" in device_name:
            device_name = "v100-pcie"
        elif "t4" in device_name:
            device_name = "t4"
        elif "quadro rtx 5000" in device_name:
            device_name = "quadro rtx 5000"
        else:
            device_name = None

        if device_name is not None:
            try:
                return int(_GPU_AVAILABLE_FLOPS[device_name][dtype])
            except KeyError:
                raise KeyError(
                    f"flop count not found for {device_name} with dtype: {dtype}; "
                    "MFU cannot be calculated and reported."
                )
    elif device.type == "xla":
        if _XLA_GREATER_EQUAL_2_1:
            from torch_xla._internal import tpu
        else:
            from torch_xla.experimental import tpu

        device_name = tpu.get_tpu_env()["TYPE"].lower()
        try:
            return int(_TPU_AVAILABLE_FLOPS[device_name])
        except KeyError:
            raise KeyError(
                f"flop count not found for {device_name} with dtype: {dtype}; MFU cannot be calculated and reported."
            )

    return None


# Adapted from https://github.com/mosaicml/composer/blob/f2a2dc820/composer/callbacks/speed_monitor.py
class _SpeedMonitorBase:
    """Logs the training throughput and utilization.

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
        - items/sec, flops/sec and MFU do not account for padding if present. We suggest using samples/sec or
          batches/sec to measure throughput under this circumstance.
        - Be careful when comparing MFU numbers across projects, as this will highly depend on the ``flops_per_batch``.
          There is no widespread, realistic, and reliable implementation to compute them.
          We suggest using our ``measure_flops`` function, but many other works will use ``estimated_flops`` which
          will almost always be an overestimate when compared to the true value.

    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Defaults to 100.
        time_unit (str, optional): Time unit to use for `time` logging. Can be one of
            'seconds', 'minutes', 'hours', or 'days'. Defaults to 'hours'.

    """

    def __init__(
        self,
        flops_available: float,
        log_dict: Callable[[Dict, int], None],
        window_size: int = 100,
        time_unit: str = "hours",
    ):
        self.flops_available = flops_available
        self.log_dict = log_dict

        # Track the batch num samples and wct to compute throughput over a window of batches
        self.history_samples: Deque[int] = deque(maxlen=window_size + 1)
        self.history_wct: Deque[float] = deque(maxlen=window_size + 1)
        self.history_lengths: Deque[int] = deque(maxlen=window_size + 1)
        self.history_flops: Deque[int] = deque(maxlen=window_size + 1)

        self.divider = 1
        if time_unit == "seconds":
            self.divider = 1
        elif time_unit == "minutes":
            self.divider = 60
        elif time_unit == "hours":
            self.divider = 60 * 60
        elif time_unit == "days":
            self.divider = 60 * 60 * 24
        else:
            raise ValueError(
                f'Invalid time_unit: {time_unit}. Must be one of "seconds", "minutes", "hours", or "days".'
            )

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0
        self.step = -1

    def on_train_batch_end(
        self,
        samples: int,  # total samples seen (per device)
        train_elapsed: float,  # total training time (seconds)
        world_size: int,
        flops_per_batch: Optional[int] = None,  # (per device)
        lengths: Optional[int] = None,  # total length of the samples seen (per device)
    ) -> None:
        self.step += 1
        step = self.step
        metrics = {}

        self.history_samples.append(samples)
        if lengths is not None:
            self.history_lengths.append(lengths)
            # if lengths are passed, there should be as many values as samples
            assert len(self.history_samples) == len(self.history_lengths)
        self.history_wct.append(train_elapsed)
        if len(self.history_wct) == self.history_wct.maxlen:
            elapsed_batches = len(self.history_samples) - 1
            elapsed_samples = self.history_samples[-1] - self.history_samples[0]
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            samples_per_sec = elapsed_samples * world_size / elapsed_wct
            dev_samples_per_sec = elapsed_samples / elapsed_wct
            metrics.update(
                {
                    "throughput/batches_per_sec": elapsed_batches * world_size / elapsed_wct,
                    "throughput/samples_per_sec": samples_per_sec,
                    "throughput/device/batches_per_sec": elapsed_batches / elapsed_wct,
                    "throughput/device/samples_per_sec": dev_samples_per_sec,
                }
            )
            if lengths is not None:
                elapsed_lengths = int(self.history_lengths[-1]) - int(self.history_lengths[0])
                avg_length = elapsed_lengths / elapsed_batches
                metrics.update(
                    {
                        "throughput/items_per_sec": samples_per_sec * avg_length,
                        "throughput/device/items_per_sec": dev_samples_per_sec * avg_length,
                    }
                )

        if flops_per_batch is not None:
            # sum of flops per batch across ranks
            self.history_flops.append(flops_per_batch * world_size)
        if len(self.history_flops) == self.history_flops.maxlen:
            elapsed_flops = sum(self.history_flops) - self.history_flops[0]
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            flops_per_sec = elapsed_flops / elapsed_wct
            device_flops_per_sec = flops_per_sec / world_size
            metrics.update(
                {"throughput/flops_per_sec": flops_per_sec, "throughput/device/flops_per_sec": device_flops_per_sec}
            )
            if self.flops_available:
                metrics["throughput/device/mfu"] = device_flops_per_sec / self.flops_available

        metrics.update(
            {
                "time/train": train_elapsed / self.divider,
                "time/val": self.total_eval_wct / self.divider,
                "time/total": (train_elapsed + self.total_eval_wct) / self.divider,
                "samples": samples,
            }
        )

        self.log_dict(metrics, step)

    def eval_end(self, eval_elapsed: float) -> None:
        self.total_eval_wct += eval_elapsed  # seconds


def _plugin_to_compute_dtype(plugin: Precision) -> torch.dtype:
    if isinstance(plugin, BitsandbytesPrecision):
        return plugin.dtype
    if isinstance(plugin, (HalfPrecision, MixedPrecision, HalfPrecisionPlugin)):
        return plugin._desired_input_dtype
    if isinstance(plugin, MixedPrecisionPlugin):
        return torch.bfloat16 if plugin.precision == "bf16-mixed" else torch.half
    if isinstance(plugin, (DoublePrecision, DoublePrecisionPlugin)):
        return torch.double
    if isinstance(plugin, (XLAPrecision, XLAPrecisionPlugin)):
        return plugin._desired_dtype
    if isinstance(plugin, TransformerEnginePrecision):
        return torch.int8
    if isinstance(plugin, (FSDPPrecision, FSDPPrecisionPlugin)):
        return plugin.mixed_precision_config.reduce_dtype
    if isinstance(plugin, Precision):
        return torch.float32
    raise NotImplementedError(plugin)


class SpeedMonitor(_SpeedMonitorBase):
    def __init__(self, fabric: Fabric, *args: Any, **kwargs: Any) -> None:
        dtype = _plugin_to_compute_dtype(fabric.strategy.precision)
        flops_available = _get_flops_available(fabric.device, dtype)
        super().__init__(flops_available, fabric.log_dict, *args, **kwargs)

    @rank_zero_only
    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        super().on_train_batch_end(*args, **kwargs)


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
        model_loss = lambda y: y.sum()
        training_flops = measure_flops(model, model_fwd, model_loss)
        eval_flops = measure_flops(model.eval(), model_fwd)

    Args:
        model: The model whose FLOPs should be measured.
        forward_fn: A function that runs ``forward`` on the model and returns the result.
        loss_fn: A function that computes the loss given the ``forward_fn`` output.

    """
    flop_counter = FlopCounterMode(model, display=False)
    ctx = nullcontext() if model.training else torch.no_grad()
    with ctx, flop_counter:
        y = forward_fn()
        if loss_fn is not None and model.training:
            loss = loss_fn(y)
            loss.backward()
    return flop_counter.get_total_flops()
