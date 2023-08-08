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
from typing import Optional

import torch

from lightning.fabric.accelerators import Accelerator
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.io.xla import XLACheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies import _StrategyRegistry
from lightning.fabric.strategies.single_device import SingleDeviceStrategy
from lightning.fabric.utilities.types import _DEVICE


class SingleDeviceXLAStrategy(SingleDeviceStrategy):
    """Strategy for training on a single XLA device."""

    def __init__(
        self,
        device: _DEVICE,
        accelerator: Optional[Accelerator] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
    ):
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        if isinstance(device, torch.device):
            # unwrap the `torch.device` in favor of `xla_device`
            device = device.index

        import torch_xla.core.xla_model as xm

        super().__init__(
            accelerator=accelerator,
            device=xm.xla_device(device),
            checkpoint_io=checkpoint_io,
            precision=precision,
        )

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = XLACheckpointIO()
        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: CheckpointIO) -> None:
        self._checkpoint_io = io

    @classmethod
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        strategy_registry.register("single_xla", cls, description=cls.__class__.__name__)
