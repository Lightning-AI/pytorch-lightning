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

import torch
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.optim import LBFGS, Optimizer
from typing_extensions import Literal

from lightning_lite.plugins.precision.precision import Precision
from lightning_lite.utilities.enums import AMPType, PrecisionType
from lightning_lite.utilities.imports import _APEX_AVAILABLE
from lightning_lite.utilities.types import Steppable

_DEEPSPEED_AVAILABLE = RequirementCache("deepspeed")
if TYPE_CHECKING and _DEEPSPEED_AVAILABLE:
    import deepspeed


class DeepSpeedPrecision(Precision):
    """Precision plugin for DeepSpeed integration.

    Args:
        precision: Full precision (32), half precision (16) or bfloat16 precision (bf16).
        amp_type: The mixed precision backend to use ("native" or "apex").
        amp_level: The optimization level to use (O1, O2, etc...). By default it will be set to "O2"
            if ``amp_type`` is set to "apex".

    Raises:
        MisconfigurationException:
            If using ``bfloat16`` precision and ``deepspeed<v0.6``.

        ValueError:
            If unsupported ``precision`` is provided.
    """

    def __init__(self, precision: Literal[16, 32, "bf16"], amp_type: str, amp_level: Optional[str] = None) -> None:
        if amp_type == AMPType.APEX:
            if not _APEX_AVAILABLE:
                raise ImportError(
                    "You have asked for Apex AMP but `apex` is not installed."
                    " Install `apex` using this guide: https://github.com/NVIDIA/apex"
                )

            amp_level = amp_level or "O2"

        supported_precision = ("16", "32", "bf16")
        if str(precision) not in supported_precision:
            raise ValueError(
                f"`precision={precision!r})` is not supported in DeepSpeed."
                f" `precision` must be one of: {', '.join(supported_precision)}."
            )

        super().__init__()
        self.precision = precision
        self.amp_type = amp_type
        self.amp_level = amp_level

    def convert_input(self, data: Tensor) -> Tensor:
        precision_to_type = {"bf16": torch.bfloat16, 16: torch.float16, 32: torch.float32}
        to_type = precision_to_type[self.precision]
        return data.to(to_type) if torch.is_floating_point(data) else data

    def backward(self, tensor: Tensor, model: "deepspeed.DeepSpeedEngine", *args: Any, **kwargs: Any) -> None:
        """Performs back-propagation using DeepSpeed's engine."""
        model.backward(tensor, *args, **kwargs)

    def optimizer_step(
        self,
        optimizer: Steppable,
        **kwargs: Any,
    ) -> Any:
        # DeepSpeed handles the optimizer step internally
        return optimizer.step(**kwargs)
