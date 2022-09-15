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
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from lightning_utilities.core.imports import module_available
from torch.distributed.constants import default_pg_timeout
from torch.nn import Module
from torch.optim import Optimizer

from lightning_lite.accelerators import Accelerator
from lightning_lite.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning_lite.strategies import DDPSpawnStrategy
from lightning_lite.strategies.ddp import DDPStrategy
from lightning_lite.utilities.enums import PrecisionType
from lightning_lite.utilities.imports import _IS_WINDOWS

_FAIRSCALE_AVAILABLE = not _IS_WINDOWS and module_available("fairscale.nn")

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS
else:
    OSS = ShardedDataParallel = object


class DDPShardedStrategy(DDPStrategy):
    """Optimizer and gradient sharded training provided by FairScale."""

    _REDUCE_BUFFER_SIZE_DEFAULT: int = 2**23  # 8M

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            process_group_backen=process_group_backend,
            timeout=timeout,
            **kwargs,
        )
        super().__init__()
        if "reduce_buffer_size" not in self._ddp_kwargs:
            # For multi-node training, enabling bucketing will improve performance.
            self._ddp_kwargs["reduce_buffer_size"] = self._REDUCE_BUFFER_SIZE_DEFAULT if self.num_nodes > 1 else 0

    def setup_module_and_optimizers(
        self, module: Module, optimizers: List[Optimizer]
    ) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model and optimizers with fairscale components.

        Return:
            The model wrapped into a :class:`~fairscale.nn.data_parallel.ShardedDataParallel` module
            and a list of optimizer wrapped in :class:~`fairscale.optim.OSS`.
        """
        optimizers = self._reinit_optimizers_with_oss(optimizers)
        model = ShardedDataParallel(module, sharded_optimizer=optimizers, **self._ddp_kwargs)
        return model, optimizers

    @contextmanager
    def block_backward_sync(self, module: Module) -> Generator:
        """Blocks syncing gradients behaviour on backwards pass.

        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(module, ShardedDataParallel):
            with module.no_sync():
                yield None
        else:
            yield None

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_sharded_find_unused_parameters_false",
            cls,
            description="DDP Sharded Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
        strategy_registry.register(
            "ddp_sharded",
            cls,
            description=cls.__class__.__name__,
        )

    def _reinit_optimizers_with_oss(self, optimizers: List[Optimizer]) -> List["OSS"]:
        for x, optimizer in enumerate(optimizers):
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
                is_fp16 = self.precision_plugin.precision in (PrecisionType.MIXED, PrecisionType.HALF)
                # For multi-node training, compressing the model shards in fp16 before broadcasting
                # improves performance. When using PyTorch AMP, it will not degrade
                # the model performance.
                zero_optimizer.broadcast_fp16 = is_fp16 and self.num_nodes > 1
                optimizers[x] = zero_optimizer
                del optimizer
        return optimizers


class DDPSpawnShardedStrategy(DDPSpawnStrategy):
    """Optimizer sharded training provided by FairScale."""

    def setup_module_and_optimizers(
        self, module: Module, optimizers: List[Optimizer]
    ) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model and optimizers with fairscale components.

        Return:
            The model wrapped into a :class:`~fairscale.nn.data_parallel.ShardedDataParallel` module
            and a list of optimizer wrapped in :class:~`fairscale.optim.OSS`.
        """
        optimizers = self._reinit_optimizers_with_oss(optimizers)
        model = ShardedDataParallel(module, sharded_optimizer=optimizers, **self._ddp_kwargs)
        return model, optimizers

    @contextmanager
    def block_backward_sync(self, module: Module) -> Generator:
        """Blocks syncing gradients behaviour on backwards pass.

        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(module, ShardedDataParallel):
            with module.no_sync():
                yield None
        else:
            yield None

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_sharded_spawn_find_unused_parameters_false",
            cls,
            description="DDP Spawn Sharded Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
        strategy_registry.register(
            "ddp_sharded_spawn",
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def _reinit_optimizers_with_oss(self, optimizers: List[Optimizer]) -> List["OSS"]:
        for x, optimizer in enumerate(optimizers):
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
                optimizers[x] = zero_optimizer
                del optimizer
        return optimizers
