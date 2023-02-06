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
from typing import Any, Dict

import torch

import pytorch_lightning as pl
from lightning_fabric.accelerators.accelerator import Accelerator as _Accelerator
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class Accelerator(_Accelerator, ABC):
    """The Accelerator base class for Lightning PyTorch.

    An Accelerator is meant to deal with one type of hardware.
    """

    def setup_environment(self, root_device: torch.device) -> None:
        """
        .. deprecated:: v1.8.0
            This hook was deprecated in v1.8.0 and will be removed in v2.0.0. Please use ``setup_device()`` instead.
        """
        rank_zero_deprecation(
            "`Accelerator.setup_environment` has been deprecated in deprecated in v1.8.0 and will be removed in"
            " v2.0.0. Please use `setup_device()` instead."
        )
        self.setup_device(root_device)

    def setup(self, trainer: "pl.Trainer") -> None:
        """Setup plugins for the trainer fit and creates optimizers.

        Args:
            trainer: the trainer instance
        """

    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Get stats for a given device.

        Args:
            device: device for which to get stats

        Returns:
            Dictionary of device stats
        """
        raise NotImplementedError
