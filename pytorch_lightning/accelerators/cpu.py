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
from __future__ import annotations

from typing import Any

import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _DEVICE


class CPUAccelerator(Accelerator):
    """Accelerator for CPU devices."""

    def setup_environment(self, root_device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not CPU.
        """
        if root_device.type != "cpu":
            raise MisconfigurationException(f"Device should be CPU, got {root_device} instead.")

    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """CPU device stats aren't supported yet."""
        return {}

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1
