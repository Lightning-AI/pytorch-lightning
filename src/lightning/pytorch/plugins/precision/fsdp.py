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
from typing import Any, Literal, Optional

import torch

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin
from lightning.pytorch.utilities.exceptions import MisconfigurationException

if _TORCH_GREATER_EQUAL_1_12 and torch.distributed.is_available():
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
else:
    MixedPrecision = None  # type: ignore[misc,assignment]
    ShardedGradScaler = None  # type: ignore[misc,assignment]


class FSDPMixedPrecisionPlugin(MixedPrecisionPlugin):
    """AMP for Fully Sharded Data Parallel (FSDP) Training."""

    def __init__(
        self, precision: Literal["16-mixed", "bf16-mixed"], device: str, scaler: Optional[ShardedGradScaler] = None
    ) -> None:
        if not _TORCH_GREATER_EQUAL_1_12:
            raise MisconfigurationException("`FSDPMixedPrecisionPlugin` is supported from PyTorch v1.12.0 onwards.")
        super().__init__(
            precision, device, scaler=(ShardedGradScaler() if scaler is None and str(precision) == "16-mixed" else None)
        )

    def clip_grad_by_norm(self, *_: Any, **__: Any) -> None:
        # see https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
        # section `Gradient Clipping`, using `torch.nn.utils.clip_grad_norm_` is incorrect
        # for FSDP module. To overcome this, needs to call sharded_module.clip_grad_norm(clip_val)
        # however we rely on LightningModule's configure_sharded_model to wrap FSDP, it would be hard to
        # trace back the root FSDP. Now we only support clip by value.
        raise MisconfigurationException(
            f"`gradient_clip_algorithm='norm'` is currently not supported for `{self.__class__.__name__}`"
        )

    @property
    def mixed_precision_config(self) -> Optional[MixedPrecision]:
        assert MixedPrecision is not None
        if self.precision == "16-mixed":
            dtype = torch.float16
        elif self.precision == "bf16-mixed":
            dtype = torch.bfloat16
        else:
            raise MisconfigurationException(f"Was unable to infer precision type, received {self.precision!r}.")
        return MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )
