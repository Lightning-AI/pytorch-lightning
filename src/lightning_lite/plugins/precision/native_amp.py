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
from typing import Any, Dict, Generator, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import LBFGS
from typing_extensions import Literal

from lightning_lite.accelerators.cuda import _patch_cuda_is_available
from lightning_lite.plugins.precision.precision import Precision
from lightning_lite.plugins.precision.utils import _convert_fp_tensor
from lightning_lite.utilities.imports import _TORCH_GREATER_EQUAL_1_10
from lightning_lite.utilities.types import Optimizable

if _TORCH_GREATER_EQUAL_1_10:
    from torch import autocast as new_autocast
else:
    from torch.cuda.amp import autocast as old_autocast


class NativeMixedPrecision(Precision):
    """Plugin for Native Mixed Precision (AMP) training with ``torch.autocast``.

    Args:
        precision: Whether to use ``torch.float16`` (``16``) or ``torch.bfloat16`` (``'bf16'``).
        device: The device for ``torch.autocast``.
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.
    """

    def __init__(
        self, precision: Literal[16, "bf16"], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__()
        if precision == "bf16" and not _TORCH_GREATER_EQUAL_1_10:
            raise ImportError("To use bfloat16 with native amp you must install torch greater or equal to 1.10.")
        if scaler is None and precision == 16:
            with _patch_cuda_is_available():
                # if possible, we defer CUDA initialization to support strategies that will attempt forks
                scaler = torch.cuda.amp.GradScaler()
        if scaler is not None and precision == "bf16":
            raise ValueError(f"`precision='bf16'` does not use a scaler, found {scaler}.")
        self.precision = precision
        self.device = device
        self.scaler = scaler

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        with self._autocast_context_manager():
            yield

    def convert_input(self, data: Tensor) -> Tensor:
        precision_to_type = {"bf16": torch.bfloat16, 16: torch.float16}
        dst_type = precision_to_type[self.precision]
        return _convert_fp_tensor(data, dst_type)

    def backward(self, tensor: Tensor, model: Optional[Module], *args: Any, **kwargs: Any) -> None:
        if self.scaler is not None:
            tensor = self.scaler.scale(tensor)
        super().backward(tensor, model, *args, **kwargs)

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> Any:
        if self.scaler is None:
            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(optimizer, **kwargs)
        if isinstance(optimizer, LBFGS):
            raise TypeError("Native AMP and the LBFGS optimizer are not compatible.")
        # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
        step_output = self.scaler.step(optimizer, **kwargs)
        self.scaler.update()
        return step_output

    def state_dict(self) -> Dict[str, Any]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)

    def _autocast_context_manager(self) -> Union["old_autocast", "new_autocast"]:
        if _TORCH_GREATER_EQUAL_1_10:
            # the dtype could be automatically inferred but we need to manually set it due to a bug upstream
            # https://github.com/pytorch/pytorch/issues/67233
            return new_autocast(self.device, dtype=torch.bfloat16 if self.precision == "bf16" else torch.half)
        return old_autocast()
