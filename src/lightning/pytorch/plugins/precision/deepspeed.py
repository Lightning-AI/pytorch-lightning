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
from typing import Any, Callable, cast, Literal, Optional, TYPE_CHECKING, Union

from torch import Tensor
from torch.optim import LBFGS, Optimizer
from typing_extensions import get_args

import lightning.pytorch as pl
from lightning.fabric.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning.fabric.utilities.types import Steppable
from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import WarningCache

if TYPE_CHECKING and _DEEPSPEED_AVAILABLE:
    import deepspeed

warning_cache = WarningCache()

_PRECISION_INPUT = Literal["32-true", "16-mixed", "bf16-mixed"]


class DeepSpeedPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for DeepSpeed integration.

    Args:
        precision: Full precision (32), half precision (16) or bfloat16 precision (bf16).
    Raises:
        ValueError:
            If unsupported ``precision`` is provided.
    """

    def __init__(self, precision: Literal["32-true", "16-mixed", "bf16-mixed"]) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`Trainer(strategy='deepspeed', precision={precision!r})` is not supported."
                f" `precision` must be one of: {supported_precision}."
            )
        self.precision = cast(_PRECISION_INPUT, str(precision))

    def backward(  # type: ignore[override]
        self,
        tensor: Tensor,
        model: "pl.LightningModule",
        optimizer: Optional[Steppable],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Performs back-propagation using DeepSpeed's engine.

        Args:
            tensor: the loss tensor
            model: the model to be optimized
            optimizer: ignored for DeepSpeed
            \*args: additional positional arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call
            \**kwargs: additional keyword arguments for the :meth:`deepspeed.DeepSpeedEngine.backward` call
        """
        if is_overridden("backward", model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since DeepSpeed handles"
                " the backward logic internally."
            )
        deepspeed_engine: "deepspeed.DeepSpeedEngine" = model.trainer.model
        deepspeed_engine.backward(tensor, *args, **kwargs)

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Steppable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException("DeepSpeed and the LBFGS optimizer are not compatible.")
        closure_result = closure()
        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if model.automatic_optimization and skipped_backward:
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not supported by `DeepSpeed`"
            )
        # DeepSpeed handles the optimizer step internally
        deepspeed_engine: "deepspeed.DeepSpeedEngine" = model.trainer.model
        return deepspeed_engine.step(**kwargs)

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        """DeepSpeed handles gradient clipping internally."""
