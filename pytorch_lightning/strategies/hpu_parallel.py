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
import os
from typing import Any, Dict, List, Optional, Union

import __main__

import torch
import torch.distributed

import pytorch_lightning as pl
from pytorch_lightning.overrides.torch_distributed import broadcast_object_list
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.io.hpu_io_plugin import HPUCheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.distributed import group as _group
from pytorch_lightning.utilities.enums import _StrategyType


class HPUParallelStrategy(DDPStrategy):
    """Plugin for multi-process single-device training on one or multiple nodes.

    The main process in each node spawns N-1 child processes via :func:`subprocess.Popen`, where N is the number of
    devices (e.g. GPU) per node. It is very similar to how :mod:`torch.distributed.launch` launches processes.
    """

    distributed_backend = _StrategyType.HPU_PARALLEL
    strategy_name = "hpu_parallel"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            checkpoint_io=checkpoint_io or HPUCheckpointIO(),
            precision_plugin=precision_plugin,
        )

    def setup_environment(self) -> None:

        import habana_frameworks.torch.core.hccl

        os.environ["ID"] = str(self.local_rank)
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "hccl"
        
        super().setup_environment()

    def broadcast(self, obj: object, src: int = 0) -> object:
        obj = [obj]
        if self.global_rank != src:
            obj = [None]
        if self.root_device.type == "hpu":
            broadcast_object_list(obj, src, group=_group.WORLD)
        else:
            torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)

        return obj[0]

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
