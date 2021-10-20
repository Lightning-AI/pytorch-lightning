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
from typing import Any, Callable, Dict, Generator, Optional, Union

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
    """Plugin for Native Mixed Precision (AMP) training with :func:`torch.autocast`.

    Args:
        precision: Whether to use torch.float16 (16) or torch.bfloat16 ('bf16').
        device: The device for `autocast`.
        scaler: An optional `GradScaler` to use.
    """

    backend = AMPType.NATIVE

    def __init__(
        self, precision: Union[str, int], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__()
        self.precision = precision
        self.device = device
        if scaler is None and self.precision == 16:
            scaler = torch.cuda.amp.GradScaler()
        self.scaler = scaler

    def pre_backward(self, model: "pl.LightningModule", closure_loss: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            closure_loss = self.scaler.scale(closure_loss)
        return super().pre_backward(model, closure_loss)

    def _run_backward(self, tensor: Tensor, model: Module, *args: Any, **kwargs: Any) -> None:
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        super()._run_backward(tensor, model, *args, **kwargs)

    def pre_optimizer_step(
        self,
        model: "pl.LightningModule",
        optimizer: Optimizer,
        optimizer_idx: int,
        lambda_closure: Callable,
        **kwargs: Any,
    ) -> bool:
        if self.scaler is None:
            return super().pre_optimizer_step(model, optimizer, optimizer_idx, lambda_closure, **kwargs)
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        result = lambda_closure()  # native amp does not support closures
        self.scaler.unscale_(optimizer)
        super().pre_optimizer_step(model, optimizer, optimizer_idx, lambda_closure, **kwargs)
        skipped_backward = result is None
        # in manual optimization, the closure does not return a value
        if not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            self.scaler.step(optimizer)
            self.scaler.update()
        return False

    def autocast_context_manager(self) -> autocast:
        if _TORCH_GREATER_EQUAL_DEV_1_10:
            return autocast(self.device)
        return autocast()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.scaler is not None and "native_amp_scaling_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["native_amp_scaling_state"])

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.scaler is not None:
            checkpoint["native_amp_scaling_state"] = self.scaler.state_dict()
