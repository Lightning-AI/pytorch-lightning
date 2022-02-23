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

import os
from typing import Any, Dict, Optional

import torch

import pytorch_lightning as pl
from pytorch_lightning.plugins.io.hpu_io_plugin import HPUCheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.plugins.precision.hpu_precision import HPUPrecisionPlugin
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _PATH


class HPUStrategy(SingleDeviceStrategy):
    """Strategy for training on HPU devices."""
    
    strategy_name = "hpu_single"

    def __init__(
        self,
        device: int,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        checkpoint_io: Optional[HPUCheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        hmp_params: Optional[str] = None,
    ):

        device = device
        checkpoint_io = checkpoint_io or HPUCheckpointIO()
        super().__init__(
            accelerator=accelerator,
            device=device, checkpoint_io=checkpoint_io, precision_plugin=precision_plugin)

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self, trainer: "pl.Trainer") -> None:
        self.model_to_device()
        super().setup(trainer)

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        super().setup_optimizers(trainer)

        if len(self.optimizers) > 1:
            raise MisconfigurationException("HPUs currently support only one optimizer.")

    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    @property
    def on_hpu(self) -> bool:
        return True

    def pre_dispatch(self) -> None:
        if isinstance(self.device, int):
            self.device = torch.device(self.device)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
