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

from typing import Any

import torch
from torch import Tensor

import lightning.pytorch as pl
from lightning.fabric.plugins import CheckpointIO
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.pytorch.strategies.strategy import Strategy, TBroadcast


class SingleDeviceStrategy(Strategy):
    """Strategy that handles communication on a single device."""

    strategy_name = "single_device"

    def __init__(
        self,
        device: _DEVICE = "cpu",
        accelerator: pl.accelerators.accelerator.Accelerator | None = None,
        checkpoint_io: CheckpointIO | None = None,
        precision_plugin: PrecisionPlugin | None = None,
    ):
        super().__init__(accelerator=accelerator, checkpoint_io=checkpoint_io, precision_plugin=precision_plugin)
        self._root_device = torch.device(device)
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    def reduce(self, tensor: Any | Tensor, *args: Any, **kwargs: Any) -> Any | Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor. As this plugin only
        operates with a single device, the reduction is simply the identity.

        Args:
            tensor: the tensor to sync and reduce
            *args: ignored
            **kwargs: ignored

        Return:
            the unmodified input as reduction is not needed for single process operation
        """
        return tensor

    def all_gather(self, tensor: Tensor, group: Any | None = None, sync_grads: bool = False) -> Tensor:
        """Perform a all_gather on all processes."""
        return tensor

    @property
    def root_device(self) -> torch.device:
        return self._root_device

    def model_to_device(self) -> None:
        assert self.model is not None, "self.model must be set before self.model.to()"
        self.model.to(self.root_device)

    def setup(self, trainer: pl.Trainer) -> None:
        self.model_to_device()
        super().setup(trainer)

    @property
    def is_global_zero(self) -> bool:
        return True

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        pass

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return obj

    @classmethod
    def register_strategies(cls, strategy_registry: dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
