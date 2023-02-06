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
from typing import Any, Callable, cast, Optional, TYPE_CHECKING, Union

from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.optim import LBFGS, Optimizer
from typing_extensions import get_args, Literal

import pytorch_lightning as pl
from lightning_fabric.utilities.types import Steppable
from pytorch_lightning.plugins.precision.apex_amp import _APEX_AVAILABLE
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, WarningCache

_DEEPSPEED_AVAILABLE = RequirementCache("deepspeed")
if TYPE_CHECKING and _DEEPSPEED_AVAILABLE:
    import deepspeed

warning_cache = WarningCache()

_PRECISION_INPUT_INT = Literal[32, 16]
_PRECISION_INPUT_STR = Literal["32", "16", "bf16"]
_PRECISION_INPUT = Union[_PRECISION_INPUT_INT, _PRECISION_INPUT_STR]


class DeepSpeedPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for DeepSpeed integration.

    Args:
        precision: Full precision (32), half precision (16) or bfloat16 precision (bf16).
    Raises:
        ValueError:
            If unsupported ``precision`` is provided.
    """

    def __init__(
        self,
        precision: Literal["32", 32, "16", 16, "bf16"],
        amp_type: Optional[str] = None,
        amp_level: Optional[str] = None,
    ) -> None:
        if amp_type == "apex":
            # TODO: remove in v2.0.0
            rank_zero_deprecation(
                "The NVIDIA/apex AMP implementation has been deprecated upstream. Consequently, its integration inside"
                " PyTorch Lightning has been deprecated in v1.9.0. Support for using it through the DeepSpeed"
                " implementation will be removed in v2.0.0."
            )
            if not _APEX_AVAILABLE:
                raise MisconfigurationException(
                    "You have asked for Apex AMP but `apex` is not installed."
                    " Install `apex` using this guide: https://github.com/NVIDIA/apex"
                )

            amp_level = amp_level or "O2"
        elif amp_level is not None:
            raise ValueError(
                f"`{type(self).__name__}(amp_level={amp_level!r})` is only relevant when using NVIDIA/apex"
            )
        if amp_type is None:
            amp_type = "native"
        else:
            rank_zero_deprecation(
                f"Passing `{type(self).__name__}(amp_type={amp_type!r})` been deprecated in v1.9.0 and will be removed"
                f" in v2.0.0. This argument is no longer necessary."
            )

        supported_precision = get_args(_PRECISION_INPUT_STR) + get_args(_PRECISION_INPUT_INT)
        if precision not in supported_precision:
            raise ValueError(
                f"`Trainer(strategy='deepspeed', precision={precision!r})` is not supported."
                f" `precision` must be one of: {supported_precision}."
            )
        self.precision = cast(_PRECISION_INPUT_STR, str(precision))
        self.amp_type = amp_type
        self.amp_level = amp_level

    def backward(  # type: ignore[override]
        self,
        tensor: Tensor,
        model: "pl.LightningModule",
        optimizer: Optional[Steppable],
        optimizer_idx: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Performs back-propagation using DeepSpeed's engine.

        Args:
            tensor: the loss tensor
            model: the model to be optimized
            optimizer: ignored for DeepSpeed
            optimizer_idx: ignored for DeepSpeed
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
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"DeepSpeed and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()
        self._after_closure(model, optimizer, optimizer_idx)
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

    def _track_grad_norm(self, trainer: "pl.Trainer") -> None:
        if trainer.track_grad_norm == -1:
            return
        # the gradients are not available in the model due to gradient partitioning in zero stage >= 2
        warning_cache.warn(
            f"You set `Trainer(track_grad_norm={trainer.track_grad_norm!r})' but this is not supported for DeepSpeed."
            " The setting will be ignored."
        )
