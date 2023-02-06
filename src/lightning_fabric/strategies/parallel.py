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
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.strategy import Strategy
from lightning_fabric.utilities.distributed import _all_gather_ddp_if_available
from lightning_fabric.utilities.types import ReduceOp


class ParallelStrategy(Strategy, ABC):
    """Strategy for training with multiple processes in parallel."""

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
    ):
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision=precision)
        self.parallel_devices = parallel_devices
        self.cluster_environment: Optional[ClusterEnvironment] = cluster_environment

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

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        """Perform a all_gather on all processes."""
        return _all_gather_ddp_if_available(tensor, group=group, sync_grads=sync_grads)

    def reduce_boolean_decision(self, decision: bool, all: bool = True) -> bool:
        """Reduces a boolean decision over distributed processes. By default is analagous to ``all`` from the
        standard library, returning ``True`` only if all input decisions evaluate to ``True``. If ``all`` is set to
        ``False``, it behaves like ``any`` instead.

        Args:
            decision: A single input decision.
            all: Whether to logically emulate ``all`` or ``any``. Defaults to True.

        Returns:
            bool: The reduced boolean decision.
        """
        decision = torch.tensor(int(decision), device=self.root_device)
        decision = self.all_reduce(decision, reduce_op=ReduceOp.SUM)
        decision = bool(decision == self.world_size) if all else bool(decision)
        return decision

    def teardown(self) -> None:
        assert self.cluster_environment is not None
        self.cluster_environment.teardown()
        return super().teardown()
