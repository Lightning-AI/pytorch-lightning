# Copyright The PyTorch Lightning team.
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
from typing import Any, Callable, Dict, Generator, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import LBFGS, Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.mixed import MixedPrecisionPlugin
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_DEV_1_10, AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TORCH_GREATER_EQUAL_DEV_1_10:
    from torch import autocast
else:
    from torch.cuda.amp import autocast


class NativeMixedPrecisionPlugin(MixedPrecisionPlugin):
    """Plugin for native mixed precision training with :mod:`torch.cuda.amp`.

    Args:
        precision: Whether to use torch.float16 (16) or torch.bfloat16 (bf16).
    """

    def __init__(self, precision: Union[int, str] = 16, use_cpu: bool = False) -> None:
        super().__init__()
        self.use_cpu = use_cpu
        self._dtype = self._select_precision_dtype(precision)
        self.backend = AMPType.NATIVE
        if not self.is_bfloat16:
            self.scaler = torch.cuda.amp.GradScaler()

    def _select_precision_dtype(self, precision: Union[int, str] = 16) -> torch.dtype:
        if precision == "bf16":
            if not _TORCH_GREATER_EQUAL_DEV_1_10:
                raise MisconfigurationException(
                    "To use bfloat16 with native amp you must install torch greater or equal to 1.10."
                )
            return torch.bfloat16
        return torch.float16

    @property
    def is_bfloat16(self) -> bool:
        return self._dtype == torch.bfloat16

    def pre_backward(self, model: "pl.LightningModule", closure_loss: torch.Tensor) -> torch.Tensor:
        if self.is_bfloat16:
            return super().pre_backward(model, closure_loss)
        closure_loss = self.scaler.scale(closure_loss)
        return super().pre_backward(model, closure_loss)

    def _run_backward(self, tensor: Tensor, model: Module, *args: Any, **kwargs: Any) -> None:
        if not self.is_bfloat16:
            tensor = self.scaler.scale(tensor)
        super()._run_backward(tensor, model, *args, **kwargs)

    def optimizer_step(
        self,
        model: Union["pl.LightningModule", Module],
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        if self.is_bfloat16:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(model, optimizer, optimizer_idx, lambda_closure, **kwargs)
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = lambda_closure()
        # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        self.scaler.unscale_(optimizer)
        if isinstance(model, pl.LightningModule):
            model.trainer.call_hook("on_before_optimizer_step", optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            self.scaler.step(optimizer, **kwargs)
            self.scaler.update()

    def autocast_context_manager(self) -> autocast:
        if _TORCH_GREATER_EQUAL_DEV_1_10:
            return autocast("cpu" if self.use_cpu else "cuda", dtype=self._dtype)
        return autocast()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "native_amp_scaling_state" in checkpoint and not self.is_bfloat16:
            self.scaler.load_state_dict(checkpoint["native_amp_scaling_state"])

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if not self.is_bfloat16:
            checkpoint["native_amp_scaling_state"] = self.scaler.state_dict()
