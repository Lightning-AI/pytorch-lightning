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
from typing import Any, Dict, List

import torch
from lightning_utilities.core.imports import package_available

from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator

_POPTORCH_AVAILABLE = package_available("poptorch")

if _POPTORCH_AVAILABLE:
    import poptorch

    _IPU_AVAILABLE = poptorch.ipuHardwareIsAvailable()
else:
    poptorch = None
    _IPU_AVAILABLE = False


class IPUAccelerator(Accelerator):
    """Accelerator for IPUs."""

    def setup_device(self, device: torch.device) -> None:
        pass

    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """IPU device stats aren't supported yet."""
        return {}

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: int) -> int:
        """Accelerator device parsing logic."""
        return devices

    @staticmethod
    def get_parallel_devices(devices: int) -> List[int]:
        """Gets parallel devices for the Accelerator."""
        return list(range(devices))

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        # TODO (@kaushikb11): 4 is the minimal unit they are shipped in.
        # Update this when api is exposed by the Graphcore team.
        return 4

    @staticmethod
    def is_available() -> bool:
        return _IPU_AVAILABLE

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "ipu",
            cls,
            description=f"{cls.__class__.__name__}",
        )
