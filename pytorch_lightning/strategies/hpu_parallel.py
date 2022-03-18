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
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.imports import _TORCH_LESSER_EQUAL_1_10_2

log = logging.getLogger(__name__)


class HPUParallelStrategy(DDPStrategy):
    """ Strategy for distributed training on multiple HPU devices """

    # The main process in each node spawns N-1 child processes via :func:`subprocess.Popen`, where N is the number of
    # devices (e.g. GPU) per node. It is very similar to how :mod:`torch.distributed.launch` launches processes.

    # Multi-device per process is not supported with habana

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

    def pre_configure_ddp(self): # type: ignore
        # if unset, default `find_unused_parameters` `True`
        # Many models require setting this parameter to True, as there are corner cases
        # when not all parameter backward hooks are fired by the autograd engine even if require_grad is set to True.
        # This flag does come with a performance hit, so it is suggested to disable in cases where it is possible.
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        if not self.lightning_module.automatic_optimization and not self._ddp_kwargs.get(
            "find_unused_parameters", False
        ):
            # TODO: PyTorch 1.7.0 DDP introduces `self.reducer._rebuild_buckets()` breaking manual_optimization
            rank_zero_warn(
                "From PyTorch 1.7.0, Lightning `manual_optimization` needs to set `find_unused_parameters=True` to"
                " properly work with DDP. Using `find_unused_parameters=True`."
            )
            self._ddp_kwargs["find_unused_parameters"] = True

        if self.root_device.type == "hpu":
            self._static_graph = False
            static_graph = self._ddp_kwargs.get("static_graph")
            if static_graph:
                # when _set_static_graph() is called find_unused_parameters does not have any significance.
                # Resetting the value of find_unused_parameters to False which is the default value to DDP
                self._ddp_kwargs["find_unused_parameters"] = False
                self._static_graph = True
            if static_graph is not None:
                # DDP does not accept static_graph as a parameter, hence removing it from the list
                del self._ddp_kwargs["static_graph"]

    def configure_ddp(self) -> None:
        # DDP does not accept static graph as param with torch < 1.11
        if _TORCH_LESSER_EQUAL_1_10_2:
            log.detail(f"{self.__class__.__name__}: configuring DistributedDataParallel")
            self.pre_configure_ddp()
            self.model = self._setup_model(LightningDistributedModule(self.model)) # type: ignore
            if self.root_device.type == "hpu" and self._static_graph:
                self._model._set_static_graph() # type: ignore
            self._register_ddp_hooks()
        else:
            self.configure_ddp()

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
