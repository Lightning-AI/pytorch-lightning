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
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Optional

from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.optim import LBFGS, Optimizer

import pytorch_lightning as pl
from lightning_fabric.utilities.types import _PARAMETERS, Optimizable
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation

_APEX_AVAILABLE = RequirementCache("apex")


def _import_amp_without_deprecation() -> ModuleType:
    # hide the warning upstream in favor of our deprecation
    warnings.filterwarnings(action="ignore", message="apex.amp is deprecated", category=FutureWarning)
    from apex import amp

    return amp


# TODO: remove in v2.0.0
class ApexMixedPrecisionPlugin(PrecisionPlugin):
    """Mixed Precision Plugin based on Nvidia/Apex (https://github.com/NVIDIA/apex)"""

    backend = "apex"

    def __init__(self, amp_level: str = "O2") -> None:
        # deprecate before the availability check so users don't install without knowning that it's deprecated
        rank_zero_deprecation(
            "The NVIDIA/apex AMP implementation has been deprecated upstream. Consequently, its integration inside"
            f" PyTorch Lightning has been deprecated in v1.9.0. The `{type(self).__name__}` class will be removed in"
            " v2.0.0. Please use PyTorch's AMP implementation available in"
            " `pytorch_lightning.plugins.MixedPrecisionPlugin` instead."
        )
        if not _APEX_AVAILABLE:
            raise MisconfigurationException(
                "You have asked for Apex AMP but `apex` is not installed."
                " Install `apex` using this guide: https://github.com/NVIDIA/apex"
            )
        super().__init__()
        self.amp_level = amp_level
        self._connected = False
        self._state_dict_loaded = False

    def main_params(self, optimizer: Optimizer) -> _PARAMETERS:
        amp = _import_amp_without_deprecation()
        return amp.master_params(optimizer)

    def dispatch(self, trainer: "pl.Trainer") -> None:
        if not self._connected:
            strategy = trainer.strategy
            amp = _import_amp_without_deprecation()
            _, strategy.optimizers = amp.initialize(
                trainer.lightning_module, strategy.optimizers, opt_level=self.amp_level
            )
            self._connected = True
        return super().dispatch(trainer)

    def backward(  # type: ignore[override]
        self,
        tensor: Tensor,
        model: "pl.LightningModule",
        optimizer: Optional[Optimizable],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r"""Run before precision plugin executes backward.

        Args:
            tensor: the loss value obtained from the closure
            model: the model to be optimized
            optimizer: current optimizer being used. ``None`` if using manual optimization
            \*args: Positional arguments intended for the actual function that performs the backward, like
                :meth:`~torch.Tensor.backward`.
            \**kwargs: Keyword arguments for the same purpose as ``*args``.
        """
        opt = optimizer or model.trainer.optimizers
        amp = _import_amp_without_deprecation()
        with amp.scale_loss(tensor, opt) as tensor:
            super().backward(tensor, model, optimizer, *args, **kwargs)

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        if self._state_dict_loaded:
            raise RuntimeError(
                "Resuming training with APEX is currently not supported. Set `amp_backend=None` for example or use a"
                " different precision plugin."
            )
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException(
                f"apex AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        closure_result = closure()
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not model.automatic_optimization or not skipped_backward:
            return optimizer.step(**kwargs)
        return closure_result

    def state_dict(self) -> Dict[str, Any]:
        amp = _import_amp_without_deprecation()
        return amp.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._state_dict_loaded = True
        return super().load_state_dict(state_dict)
