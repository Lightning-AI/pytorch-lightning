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
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_10, AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TORCH_GREATER_EQUAL_1_10:
    from torch import autocast as new_autocast
else:
    from torch.cuda.amp import autocast as old_autocast


class NativeMixedPrecisionPlugin(MixedPrecisionPlugin):
    """Plugin for Native Mixed Precision (AMP) training with ``torch.autocast``.

    Args:
        precision: Whether to use ``torch.float16`` (``16``) or ``torch.bfloat16`` (``'bf16'``).
        device: The device for ``torch.autocast``.
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.
    """

    backend = AMPType.NATIVE

    def __init__(
        self, precision: Union[str, int], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__()
        if precision == "bf16" and not _TORCH_GREATER_EQUAL_1_10:
            raise MisconfigurationException(
                "To use bfloat16 with native amp you must install torch greater or equal to 1.10."
            )
        if scaler is None and precision == 16:
            scaler = torch.cuda.amp.GradScaler()
        if scaler is not None and precision == "bf16":
            raise MisconfigurationException(f"`precision='bf16'` does not use a scaler, found {scaler}.")
        self.precision = precision
        self.device = device
        self.scaler = scaler

    def pre_backward(self, model: "pl.LightningModule", closure_loss: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            closure_loss = self.scaler.scale(closure_loss)
        return super().pre_backward(model, closure_loss)

    def _run_backward(self, tensor: Tensor, model: Optional[Module], *args: Any, **kwargs: Any) -> None:
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        super()._run_backward(tensor, model, *args, **kwargs)

    def optimizer_step(
        self,
        model: Union["pl.LightningModule", Module],
        optimizer: Optimizer,
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(model, optimizer, optimizer_idx, closure, **kwargs)
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()
        # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        self.scaler.unscale_(optimizer)
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)
            self.scaler.update()
            return step_output
        return closure_result

    def autocast_context_manager(self) -> Union["old_autocast", "new_autocast"]:
        if _TORCH_GREATER_EQUAL_1_10:
            # the dtype could be automatically inferred but we need to manually set it due to a bug upstream
            # https://github.com/pytorch/pytorch/issues/67233
            return new_autocast(self.device, dtype=torch.bfloat16 if self.precision == "bf16" else torch.half)
        return old_autocast()

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    def state_dict(self) -> Dict[str, Any]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)
