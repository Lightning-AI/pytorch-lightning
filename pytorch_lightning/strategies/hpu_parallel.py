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
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.distributed

import pytorch_lightning as pl
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.torch_distributed import broadcast_object_list
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.io.hpu_plugin import HPUCheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities.distributed import group as _group
from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)


class HPUParallelStrategy(DDPStrategy):
    """Plugin for multi-process single-device training on one or multiple nodes.

    The main process in each node spawns N-1 child processes via :func:`subprocess.Popen`, where N is the number of
    devices (e.g. GPU) per node. It is very similar to how :mod:`torch.distributed.launch` launches processes.
    """

    strategy_name = "hpu_parallel"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ) -> None:

        if not _HPU_AVAILABLE:
            raise MisconfigurationException("`HPUParallelStrategy` requires HPU devices to run")

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            checkpoint_io=checkpoint_io or HPUCheckpointIO(),
            precision_plugin=precision_plugin,
        )

    def setup_environment(self) -> None:

        from habana_frameworks.torch.utils.library_loader import load_habana_module

        load_habana_module()
        import habana_frameworks.torch.core.hccl  # noqa: F401

        os.environ["ID"] = str(self.local_rank)
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "hccl"

        super().setup_environment()

    def determine_ddp_device_ids(self) -> None:
        return None

    def configure_ddp(self) -> None:
        log.detail(f"{self.__class__.__name__}: configuring DistributedDataParallel")
        self.pre_configure_ddp()
        self.model = self._setup_model(LightningDistributedModule(self.model))
        if self.root_device.type == "hpu" and self._static_graph:
            self._model._set_static_graph()
        self._register_ddp_hooks()

    def broadcast(self, obj: object, src: int = 0) -> object:  # type: ignore
        obj = [obj]
        if self.global_rank != src:
            obj = [None]

        broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
