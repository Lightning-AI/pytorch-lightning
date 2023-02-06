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
from typing import Any, Callable, cast, Optional, Union

from torch import Tensor
from torch.optim import Optimizer
from typing_extensions import Literal

import pytorch_lightning as pl
from lightning_fabric.utilities.types import Steppable
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.rank_zero import WarningCache

warning_cache = WarningCache()


class ColossalAIPrecisionPlugin(PrecisionPlugin):
    """Precision plugin for ColossalAI integration.

    Args:
        precision: Half precision (16).

    Raises:
        ValueError:
            If precison is not 16.
    """

    def __init__(self, precision: Literal["16", 16] = 16) -> None:
        if precision not in ("16", 16):
            raise ValueError(
                f"`Trainer(strategy='colossalai', precision={precision!r})` is not supported."
                " Consider setting `precision=16`."
            )
        self.precision = cast(Literal["16"], str(precision))

    def backward(  # type: ignore[override]
        self,
        tensor: Tensor,
        model: "pl.LightningModule",
        optimizer: Optional[Steppable],
        optimizer_idx: Optional[int],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        assert optimizer is not None
        optimizer.backward(tensor)

    def clip_grad_by_norm(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        optimizer.clip_grad_norm(None, clip_val)

    def clip_grad_by_value(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        raise NotImplementedError("`clip_grad_by_value` is not supported by `ColossalAI`")

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Steppable,
        model: "pl.LightningModule",
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        closure_result = closure()
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        if isinstance(model, pl.LightningModule) and model.automatic_optimization and skipped_backward:
            raise ValueError(
                "Skipping backward by returning `None` from your `training_step` is not supported by `ColossalAI`."
            )
        optimizer.step()

    def _track_grad_norm(self, trainer: "pl.Trainer") -> None:
        if trainer.track_grad_norm == -1:
            return
        # the gradients are not available in the model due to gradient partitioning in zero stage >= 2
        warning_cache.warn(
            f"You set `Trainer(track_grad_norm={trainer.track_grad_norm!r})' but this is not supported for ColossalAI."
            " The setting will be ignored."
        )
