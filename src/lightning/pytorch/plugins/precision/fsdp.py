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
from contextlib import contextmanager
from typing import Any, Generator, Literal, Optional, TYPE_CHECKING

import torch

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin
from lightning.pytorch.utilities.exceptions import MisconfigurationException

if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as TorchMixedPrecision
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


class FSDPMixedPrecisionPlugin(MixedPrecisionPlugin):
    """AMP for Fully Sharded Data Parallel (FSDP) Training.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(
        self, precision: Literal["16-mixed", "bf16-mixed"], device: str, scaler: Optional["ShardedGradScaler"] = None
    ) -> None:
        if not _TORCH_GREATER_EQUAL_1_12:
            raise MisconfigurationException("`FSDPMixedPrecisionPlugin` is supported from PyTorch v1.12.0 onwards.")
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        super().__init__(
            precision, device, scaler=(ShardedGradScaler() if scaler is None and str(precision) == "16-mixed" else None)
        )

    def clip_grad_by_norm(self, *_: Any, **__: Any) -> None:
        # see https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
        # section `Gradient Clipping`, using `torch.nn.utils.clip_grad_norm_` is incorrect with FSDP.
        # To overcome this we need to call root_sharded_module.clip_grad_norm(clip_val), but we don't have a reference
        # to the root module
        raise MisconfigurationException(
            f"`gradient_clip_algorithm='norm'` is currently not supported for `{self.__class__.__name__}`"
        )

    @property
    def mixed_precision_config(self) -> "TorchMixedPrecision":
        from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision as TorchMixedPrecision

        if self.precision == "16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.float16
        elif self.precision == "bf16-mixed":
            param_dtype = torch.float32
            reduce_dtype = buffer_dtype = torch.bfloat16
        elif self.precision == "16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.float16
        elif self.precision == "bf16-true":
            param_dtype = reduce_dtype = buffer_dtype = torch.bfloat16
        else:
            raise MisconfigurationException(f"Was unable to infer precision type, received {self.precision!r}.")

        return TorchMixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

    @contextmanager
    def init_context(self) -> Generator[None, None, None]:
        """A context manager to change the default tensor type when initializing module parameters or tensors.

        See: :meth:`torch.set_default_dtype`

        """
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.mixed_precision_config.param_dtype)
        yield
        torch.set_default_dtype(default_dtype)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """For FSDP, this context manager is a no-op since conversion is already handled internally.

        See: https://pytorch.org/docs/stable/fsdp.html for more details on mixed precision.

        """
        yield
