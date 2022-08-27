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

from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers, LightningOptimizer
from pytorch_lightning.utilities import rank_zero_deprecation


class TrainerOptimizersMixin(ABC):
    r"""
    .. deprecated:: v1.6
        The `TrainerOptimizersMixin` was deprecated in v1.6 and will be removed in v1.8.
    """

    def init_optimizers(self, model: Optional["pl.LightningModule"]) -> Tuple[List, List, List]:
        r"""
        .. deprecated:: v1.6
            `TrainerOptimizersMixin.init_optimizers` was deprecated in v1.6 and will be removed in v1.8.
        """
        rank_zero_deprecation(
            "`TrainerOptimizersMixin.init_optimizers` was deprecated in v1.6 and will be removed in v1.8."
        )
        pl_module = self.lightning_module or model
        assert isinstance(pl_module, pl.LightningModule)
        return _init_optimizers_and_lr_schedulers(pl_module)

    def convert_to_lightning_optimizers(self) -> None:
        r"""
        .. deprecated:: v1.6
            `TrainerOptimizersMixin.convert_to_lightning_optimizers` was deprecated in v1.6 and will be removed in v1.8.
        """
        rank_zero_deprecation(
            "`TrainerOptimizersMixin.convert_to_lightning_optimizers` was deprecated in v1.6 and will be removed in "
            "v1.8."
        )

        def _convert_to_lightning_optimizer(optimizer: Optimizer) -> LightningOptimizer:
            if not isinstance(optimizer, LightningOptimizer):
                optimizer = LightningOptimizer(optimizer)  # type: ignore [assignment]
            optimizer._trainer = self
            for opt_idx, opt in enumerate(self.optimizers):
                if opt == optimizer._optimizer:
                    optimizer._optimizer_idx = opt_idx
                    break
            return optimizer  # type: ignore [return-value]

        self.strategy._cached_lightning_optimizers = {
            idx: _convert_to_lightning_optimizer(opt) for idx, opt in enumerate(self.optimizers)
        }
