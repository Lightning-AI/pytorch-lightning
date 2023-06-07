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
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator

import torch
from torch import Tensor

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.utilities.distributed import _all_gather_ddp_if_available, ReduceOp
from lightning.pytorch.plugins import LayerSync
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.pytorch.strategies.strategy import Strategy


class ParallelStrategy(Strategy, ABC):
    """Plugin for training with multiple processes in parallel."""

    def __init__(
        self,
        accelerator: pl.accelerators.Accelerator | None = None,
        parallel_devices: list[torch.device] | None = None,
        cluster_environment: ClusterEnvironment | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: PrecisionPlugin | None = None,
    ):
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision_plugin=precision_plugin)
        self.parallel_devices = parallel_devices
        self.cluster_environment: ClusterEnvironment | None = cluster_environment
        self._layer_sync: LayerSync | None = None

    @property
    @abstractmethod
    def root_device(self) -> torch.device:
        """Return the root device."""

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
    def parallel_devices(self) -> list[torch.device] | None:
        return self._parallel_devices

    @parallel_devices.setter
    def parallel_devices(self, parallel_devices: list[torch.device] | None) -> None:
        self._parallel_devices = parallel_devices

    @property
    def distributed_sampler_kwargs(self) -> dict[str, Any]:
        return {
            "num_replicas": len(self.parallel_devices) if self.parallel_devices is not None else 0,
            "rank": self.global_rank,
        }

    def all_gather(self, tensor: Tensor, group: Any | None = None, sync_grads: bool = False) -> Tensor:
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
        decision = self.reduce(
            decision,
            reduce_op=ReduceOp.SUM,  # type: ignore[arg-type]
        )
        decision = bool(decision == self.world_size) if all else bool(decision)
        return decision

    @contextmanager
    def block_backward_sync(self) -> Generator:
        """Blocks ddp sync gradients behaviour on backwards pass.

        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(self.model, pl.utilities.types.DistributedDataParallel):
            with self.model.no_sync():
                yield None
        else:
            yield None

    def teardown(self) -> None:
        assert self.cluster_environment is not None
        self.cluster_environment.teardown()
        super().teardown()
