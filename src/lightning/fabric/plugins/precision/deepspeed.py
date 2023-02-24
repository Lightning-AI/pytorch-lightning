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
from typing import Any, Literal, TYPE_CHECKING

import torch
from torch import Tensor
from typing_extensions import get_args

from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.utils import _convert_fp_tensor
from lightning.fabric.utilities.types import Steppable

if TYPE_CHECKING:
    from lightning.fabric.strategies.deepspeed import _DEEPSPEED_AVAILABLE

    if _DEEPSPEED_AVAILABLE:  # type: ignore[has-type]
        import deepspeed

_PRECISION_INPUT = Literal["32-true", "16-mixed", "bf16-mixed"]


class DeepSpeedPrecision(Precision):
    """Precision plugin for DeepSpeed integration.

    Args:
        precision: Full precision (32-true), half precision (16-mixed) or bfloat16 precision (bf16-mixed).

    Raises:
        ValueError:
            If unsupported ``precision`` is provided.
    """

    def __init__(self, precision: _PRECISION_INPUT) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`precision={precision!r})` is not supported in DeepSpeed."
                f" `precision` must be one of: {supported_precision}."
            )
        self.precision = precision

    def convert_input(self, data: Tensor) -> Tensor:
        precision_to_type = {"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16, "32-true": torch.float32}
        dst_type = precision_to_type[self.precision]
        return _convert_fp_tensor(data, dst_type)

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
