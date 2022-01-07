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
from typing import Any, Dict, Union

import torch

import pytorch_lightning as pl


class Accelerator:
    """The Accelerator Base Class. An Accelerator is meant to deal with one type of Hardware.

    Currently there are accelerators for:

    - CPU
    - GPU
    - TPU
    - IPU
    """

    def setup_environment(self, root_device: torch.device) -> None:
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

    @staticmethod
    @abstractmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
