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
from torch.optim import LBFGS, Optimizer

import pytorch_lightning as pl
from lightning_lite.accelerators.cuda import _patch_cuda_is_available
from lightning_lite.utilities.types import Optimizable
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import _TORCH_GREATER_EQUAL_1_10, AMPType, GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TORCH_GREATER_EQUAL_1_10:
    from torch import autocast as new_autocast
else:
    from torch.cuda.amp import autocast as old_autocast


class NativeMixedPrecisionPlugin(PrecisionPlugin):
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
            with _patch_cuda_is_available():
                # if possible, we defer CUDA initialization to support strategies that will attempt forks
                scaler = torch.cuda.amp.GradScaler()
        if scaler is not None and precision == "bf16":
            raise MisconfigurationException(f"`precision='bf16'` does not use a scaler, found {scaler}.")
        self.precision = precision
        self.device = device
        self.scaler = scaler

    def pre_backward(self, tensor: Tensor, module: "pl.LightningModule") -> Tensor:  # type: ignore[override]
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        return super().pre_backward(tensor, module)

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(
                optimizer, model=model, optimizer_idx=optimizer_idx, closure=closure, **kwargs
            )
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()

        if not _optimizer_handles_unscaling(optimizer):
            # Unscaling needs to be performed here in case we are going to apply gradient clipping.
            # Optimizers that perform unscaling in their `.step()` method are not supported (e.g., fused Adam).
            # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
            self.scaler.unscale_(optimizer)

        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            step_output = self.scaler.step(optimizer, **kwargs)
            self.scaler.update()
            return step_output
        return closure_result

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        if clip_val > 0 and _optimizer_handles_unscaling(optimizer):
            raise RuntimeError(
                f"The current optimizer, {type(optimizer).__qualname__}, does not allow for gradient clipping"
                " because it performs unscaling of gradients internally. HINT: Are you using a 'fused' optimizer?"
            )
        super().clip_gradients(optimizer=optimizer, clip_val=clip_val, gradient_clip_algorithm=gradient_clip_algorithm)

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


def _optimizer_handles_unscaling(optimizer: Any) -> bool:
    """Determines whether a PyTorch optimizer handles unscaling gradients in the step method rather than through the
    :class:`torch.cuda.amp.GradScaler`.

    Since, the current implementation of this function checks a PyTorch internal variable on the optimizer, the return
    value will only be reliable for built-in PyTorch optimizers.
    """
    return getattr(optimizer, "_step_supports_amp_scaling", False)
