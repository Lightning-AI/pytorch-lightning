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
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.plugins.collectives import Collective, TorchCollective
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies.strategy import Strategy, TBroadcast
from lightning.fabric.utilities.distributed import _all_gather_if_available, _all_reduce_if_available
from lightning.fabric.utilities.types import ReduceOp


class ParallelStrategy(Strategy, ABC):
    """Strategy for training with multiple processes in parallel."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        collective: Optional[Collective] = None,
    ):
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision=precision)
        self.parallel_devices = parallel_devices
        self.cluster_environment: Optional[ClusterEnvironment] = cluster_environment
        self.collective: Collective = collective if collective is not None else TorchCollective()

    @property
    def global_rank(self) -> int:
        return self.cluster_environment.global_rank() if self.cluster_environment is not None else 0

    @property
    def local_rank(self) -> int:
        return self.cluster_environment.local_rank() if self.cluster_environment is not None else 0

    @property
    def node_rank(self) -> int:
        return self.cluster_environment.node_rank() if self.cluster_environment is not None else 0

    @property
    def world_size(self) -> int:
        return self.cluster_environment.world_size() if self.cluster_environment is not None else 1

    @property
    @override
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def parallel_devices(self) -> Optional[List[torch.device]]:
        return self._parallel_devices

    @parallel_devices.setter
    def parallel_devices(self, parallel_devices: Optional[List[torch.device]]) -> None:
        self._parallel_devices = parallel_devices

    @property
    def distributed_sampler_kwargs(self) -> Optional[Dict[str, Any]]:
        """Arguments for the ``DistributedSampler``.

        If this method is not defined, or it returns ``None``, then the ``DistributedSampler`` will not be used.

        """
        return {"num_replicas": self.world_size, "rank": self.global_rank}

    @override
    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        return _all_gather_if_available(tensor, collective=self.collective, sync_grads=sync_grads)

    @override
    def all_reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        return _all_reduce_if_available(tensor, collective=self.collective, reduce_op=reduce_op)

    @override
    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not self.collective.is_initialized():
            return
        self.collective.barrier(device_ids=([self.root_device.index] if self.root_device.index is not None else None))

    @override
    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not self.collective.is_initialized():
            return obj

        object_list = [obj]
        self.collective.broadcast_object_list(object_list=object_list, src=src, device=self.root_device)
        return object_list[0]

    @override
    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        """Reduces a boolean decision over distributed processes. By default is analagous to ``all`` from the standard
        library, returning ``True`` only if all input decisions evaluate to ``True``. If ``all`` is set to ``False``,
        it behaves like ``any`` instead.

        Args:
            decision: A single input decision.
            all: Whether to logically emulate ``all`` or ``any``. Defaults to True.

        Returns:
            bool: The reduced boolean decision.

        """
        decision = torch.tensor(int(decision), device=self.root_device)
        decision = self.all_reduce(
            decision,
            reduce_op=ReduceOp.SUM,  # type: ignore[arg-type]
        )
        decision = bool(decision == self.world_size) if all else bool(decision)
        return decision

    @override
    def teardown(self) -> None:
        assert self.cluster_environment is not None
        self.cluster_environment.teardown()
        self.collective.teardown()  # TODO: is this desired?
        return super().teardown()
