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
from typing import Any, Optional, TYPE_CHECKING, Union

from torch import Tensor
from torch.optim import LBFGS, Optimizer

from lightning_lite.lite.plugins.precision.precision import PrecisionPlugin
from lightning_lite.lite.utilities.enums import AMPType, PrecisionType
from lightning_lite.lite.utilities.imports import _APEX_AVAILABLE, _RequirementAvailable

_DEEPSPEED_AVAILABLE = _RequirementAvailable("deepspeed")
if TYPE_CHECKING and _DEEPSPEED_AVAILABLE:
    import deepspeed


class DeepSpeedPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for DeepSpeed integration.

    Args:
        precision: Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16).
        amp_type: The mixed precision backend to use ("native" or "apex").
        amp_level: The optimization level to use (O1, O2, etc...). By default it will be set to "O2"
            if ``amp_type`` is set to "apex".

    Raises:
        MisconfigurationException:
            If using ``bfloat16`` precision and ``deepspeed<v0.6``.

        ValueError:
            If unsupported ``precision`` is provided.
    """

    def __init__(self, precision: Union[str, int], amp_type: str, amp_level: Optional[str] = None) -> None:
        if amp_type == AMPType.APEX:
            if not _APEX_AVAILABLE:
                raise ImportError(
                    "You have asked for Apex AMP but `apex` is not installed."
                    " Install `apex` using this guide: https://github.com/NVIDIA/apex"
                )

            amp_level = amp_level or "O2"

        supported_precision = (PrecisionType.HALF, PrecisionType.FLOAT, PrecisionType.BFLOAT)
        if precision not in supported_precision:
            raise ValueError(
                f"`Trainer(strategy='deepspeed', precision={precision!r})` is not supported."
                f" `precision` must be one of: {(x.value for x in supported_precision)}."
            )

        super().__init__()
        self.precision = precision
        self.amp_type = amp_type
        self.amp_level = amp_level

    def backward(self, tensor: Tensor, model: Optional["deepspeed.DeepSpeedEngine"], *args: Any, **kwargs: Any) -> None:
        """Performs back-propagation using DeepSpeed's engine."""
        if model is None:
            raise ValueError("Please provide the model as input to `backward`.")
        model.backward(tensor, *args, **kwargs)

    def optimizer_step(
        self,
        optimizer: Optimizer,
        *args: Any,
        model: Optional["deepspeed.DeepSpeedEngine"] = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(optimizer, LBFGS):
            raise TypeError(f"DeepSpeed and the LBFGS optimizer are not compatible.")
        # DeepSpeed handles the optimizer step internally
        return model.step(**kwargs)
