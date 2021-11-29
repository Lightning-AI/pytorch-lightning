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
from typing import Any, Dict, Union

import torch

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class IPUAccelerator(Accelerator):
    """Accelerator for IPUs."""

    def setup_optimizers(self, trainer: "pl.Trainer") -> None:
        """
        Raises:
            MisconfigurationException:
                If multiple optimizers are provided.
        """
        super().setup_optimizers(trainer)

        if len(self.optimizers) > 1:
            raise MisconfigurationException("IPUs currently only support one optimizer.")

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """IPU device stats aren't supported yet."""
        return {}

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        # TODO (@kaushikb11): 4 is the minimal unit they are shipped in.
        # Update this when api is exposed by the Graphcore team.
        return 4
