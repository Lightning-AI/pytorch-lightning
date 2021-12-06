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
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import Module

import pytorch_lightning as pl
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.plugins.training_type import TrainingTypePlugin


class Accelerator:
    """The Accelerator Base Class. An Accelerator is meant to deal with one type of Hardware.

    Currently there are accelerators for:

    - CPU
    - GPU
    - TPU
    - IPU
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def root_device(self):
        return None

    def setup_environment(self) -> None:
        """Setup any processes or distributed connections.

        This is called before the LightningModule/DataModule setup hook which allows the user to access the accelerator
        environment before setup is complete.
        """

    def setup(self, trainer: "pl.Trainer") -> None:
        """Setup plugins for the trainer fit and creates optimizers.

        Args:
            trainer: the trainer instance
        """

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Gets stats for a given device.

        Args:
            device: device for which to get stats

        Returns:
            Dictionary of device stats
        """
        raise NotImplementedError

    def on_train_start(self) -> None:
        """Called when train begins."""

    @staticmethod
    @abstractmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
