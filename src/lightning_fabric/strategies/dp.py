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
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import DataParallel, Module

from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.parallel import ParallelStrategy
from lightning_fabric.strategies.strategy import TBroadcast, TReduce
from lightning_fabric.utilities.apply_func import apply_to_collection
from lightning_fabric.utilities.distributed import ReduceOp


class DataParallelStrategy(ParallelStrategy):
    """Implements data-parallel training in a single process, i.e., the model gets replicated to each device and
    each gets a split of the data."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
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
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[0]

    @property
    def distributed_sampler_kwargs(self) -> None:
        return None

    def setup_module(self, module: Module) -> DataParallel:
        """Wraps the given model into a :class:`~torch.nn.parallel.DataParallel` module."""
        return DataParallel(module=module, device_ids=self.parallel_devices)

    def module_to_device(self, module: Module) -> None:
        module.to(self.root_device)

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        # DataParallel handles the transfer of batch to the device
        return batch

    def all_reduce(
        self, collection: TReduce, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> TReduce:
        def mean(t: Tensor) -> Tensor:
            original_dtype = t.dtype
            return t.float().mean().to(original_dtype)

        return apply_to_collection(collection, Tensor, mean)

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        pass

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return obj

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        return decision

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register("dp", cls, description=cls.__class__.__name__)
