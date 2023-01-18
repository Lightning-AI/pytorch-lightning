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
from typing import List, Optional, Union

import torch.nn as nn
from lightning_utilities.core.imports import package_available
from torch.optim import Optimizer

import pytorch_lightning as pl
from lightning_fabric.plugins import Precision
from lightning_fabric.utilities.imports import _IS_WINDOWS
from pytorch_lightning.overrides.base import (
    _LightningModuleWrapperBase,
    _LightningPrecisionModuleWrapperBase,
    unwrap_lightning_module,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation

_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and package_available("fairscale")

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS
else:
    OSS = object


class LightningShardedDataParallel(_LightningModuleWrapperBase):
    def __init__(
        self,
        forward_module: Optional[Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]] = None,
        pl_module: Optional[Union["pl.LightningModule", _LightningPrecisionModuleWrapperBase]] = None,
    ) -> None:
        rank_zero_deprecation(
            "PyTorch Lightning's sharded implementation using FairScale has been deprecated in v1.9.0 and will be"
            " removed in v2.0.0. You can try using the `Trainer(strategy='fsdp_native')` instead."
            " The difference is that native FSDP uses PyTorch's implementation and the current strategy uses"
            " FairScale's implementation (which was upstreamed to PyTorch). After removal, `strategy='fsdp'` will use"
            " the native version by default."
        )
        self._validate_init_arguments(pl_module, forward_module)
        super().__init__(forward_module=(pl_module or forward_module))


def unwrap_lightning_module_sharded(wrapped_model: nn.Module) -> "pl.LightningModule":
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel

    rank_zero_deprecation(
        "The function `unwrap_lightning_module_sharded` is deprecated in v1.8.0 and will be removed in v2.0.0."
        " Access the `LightningModule` directly through the strategy attribute `Strategy.lightning_module`."
    )
    model = wrapped_model
    if isinstance(model, ShardedDataParallel):
        model = model.module

    return unwrap_lightning_module(model, _suppress_warning=True)


def _reinit_optimizers_with_oss(optimizers: List[Optimizer], precision: Precision, num_nodes: int) -> List["OSS"]:
    for x, optimizer in enumerate(optimizers):
        if not isinstance(optimizer, OSS):
            optim_class = type(optimizer)
            zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
            is_fp16 = precision.precision == "16"
            # For multi-node training, compressing the model shards in fp16 before broadcasting
            # improves performance. When using PyTorch AMP, it will not degrade
            # the model performance.
            zero_optimizer.broadcast_fp16 = is_fp16 and num_nodes > 1
            optimizers[x] = zero_optimizer
            del optimizer
    return optimizers
