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

from abc import ABC
from typing import List, Optional, Tuple
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Tuple, Union

import torch
from torch import optim
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import (
    _convert_to_lightning_optimizers,
    _init_optimizers_and_lr_schedulers,
    LightningOptimizer,
)
from pytorch_lightning.utilities import rank_zero_deprecation


@runtime_checkable
class _SupportedLRScheduler(Protocol):
    """This class is used to detect if an object is stateful using `isinstance(obj, _SupportedLRScheduler)`"""

    def step(self, *args: Any, **kwargs: Any) -> None:
        ...

    def state_dict(self) -> Dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...


class TrainerOptimizersMixin(ABC):
    r"""
    .. deprecated:: v1.6
        The `TrainerOptimizersMixin` was deprecated in v1.6 and will be removed in v1.8.
    """

    _lightning_optimizers: Optional[List[LightningOptimizer]]

    def init_optimizers(self, model: Optional["pl.LightningModule"]) -> Tuple[List, List, List]:
        r"""
        .. deprecated:: v1.6
            `TrainerOptimizersMixin.init_optimizers` was deprecated in v1.6 and will be removed in v1.8.
        """
        rank_zero_deprecation(
            "`TrainerOptimizersMixin.init_optimizers` was deprecated in v1.6 and will be removed in v1.8."
        )
        pl_module = self.lightning_module or model
        return _init_optimizers_and_lr_schedulers(pl_module)

    def convert_to_lightning_optimizers(self):
        r"""
        .. deprecated:: v1.6
            `TrainerOptimizersMixin.convert_to_lightning_optimizers` was deprecated in v1.6 and will be removed in v1.8.
        """
        rank_zero_deprecation(
            "`TrainerOptimizersMixin.convert_to_lightning_optimizers` was deprecated in v1.6 and will be removed in "
            "v1.8."
        )
        _convert_to_lightning_optimizers(self)
