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
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.nn import DataParallel, Module
from typing_extensions import override

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.strategies.registry import _StrategyRegistry
from lightning.fabric.strategies.strategy import TBroadcast, TReduce
from lightning.fabric.utilities.apply_func import apply_to_collection
from lightning.fabric.utilities.distributed import ReduceOp


class DataParallelStrategy(ParallelStrategy):
    """Implements data-parallel training in a single process, i.e., the model gets replicated to each device and each
    gets a split of the data."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
    ):
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=None,
            checkpoint_io=checkpoint_io,
            precision=precision,
        )

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[0]

    @property
    @override
    def distributed_sampler_kwargs(self) -> None:
        return None

    @override
    def setup_module(self, module: Module) -> DataParallel:
        """Wraps the given model into a :class:`~torch.nn.DataParallel` module."""
        return DataParallel(module=module, device_ids=self.parallel_devices)

    @override
    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

    @override
    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        # DataParallel handles the transfer of batch to the device
        return batch

    @override
    def all_reduce(
        self, collection: TReduce, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> TReduce:
        def mean(t: Tensor) -> Tensor:
            original_dtype = t.dtype
            return t.float().mean().to(original_dtype)

        return apply_to_collection(collection, Tensor, mean)

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        pass

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return obj

    @override
    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        return decision

    @override
    def get_module_state_dict(self, module: Module) -> dict[str, Union[Any, Tensor]]:
        if isinstance(module, DataParallel):
            module = module.module
        return super().get_module_state_dict(module)

    @override
    def load_module_state_dict(
        self, module: Module, state_dict: dict[str, Union[Any, Tensor]], strict: bool = True
    ) -> None:
        if isinstance(module, DataParallel):
            module = module.module
        super().load_module_state_dict(module=module, state_dict=state_dict, strict=strict)

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("dp", cls, description=cls.__name__)
