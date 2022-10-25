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
from typing import Optional, Union

import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import (
    _LightningModuleWrapperBase,
    _LightningPrecisionModuleWrapperBase,
    unwrap_lightning_module,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class LightningShardedDataParallel(_LightningModuleWrapperBase):
    def __init__(
        self,
        forward_module: Optional[Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]] = None,
        pl_module: Optional[Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]] = None,
    ) -> None:
        self._validate_init_arguments(pl_module, forward_module)
        super().__init__(forward_module=(pl_module or forward_module))


def unwrap_lightning_module_sharded(wrapped_model: nn.Module) -> "pl.LightningModule":
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel

    rank_zero_deprecation(
        "The function `unwrap_lightning_module_sharded` is deprecated in v1.8.0 and will be removed in v1.10.0."
        " Access the `LightningModule` directly through the strategy attribute `Strategy.lightning_module`."
    )
    model = wrapped_model
    if isinstance(model, ShardedDataParallel):
        model = model.module

    return unwrap_lightning_module(model, _suppress_warning=True)
