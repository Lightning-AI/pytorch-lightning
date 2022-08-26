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

from typing import Any, Dict, List, Optional, Union

import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.device_parser import parse_hpus
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _HPU_AVAILABLE
from pytorch_lightning.utilities.rank_zero import rank_zero_debug

if _HPU_AVAILABLE:
    import habana_frameworks.torch.hpu as torch_hpu


class HPUAccelerator(Accelerator):
    """Accelerator for HPU devices."""

    def setup_environment(self, root_device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not HPU.
        """
        super().setup_environment(root_device)
        if root_device.type != "hpu":
            raise MisconfigurationException(f"Device should be HPU, got {root_device} instead.")

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Returns a map of the following metrics with their values:

        - Limit: amount of total memory on HPU device.
        - InUse: amount of allocated memory at any instance.
        - MaxInUse: amount of total active memory allocated.
        - NumAllocs: number of allocations.
        - NumFrees: number of freed chunks.
        - ActiveAllocs: number of active allocations.
        - MaxAllocSize: maximum allocated size.
        - TotalSystemAllocs: total number of system allocations.
        - TotalSystemFrees: total number of system frees.
        - TotalActiveAllocs: total number of active allocations.
        """
        try:
            return torch_hpu.hpu.memory_stats(device)
        except (AttributeError, NameError):
            rank_zero_debug("HPU `get_device_stats` failed")
            return {}

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[int]:
        """Accelerator device parsing logic."""
        return parse_hpus(devices)

    @staticmethod
    def get_parallel_devices(devices: int) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("hpu")] * devices

    @staticmethod
    def auto_device_count() -> int:
        """Returns the number of HPU devices when the devices is set to auto."""
        try:
            return torch_hpu.device_count()
        except (AttributeError, NameError):
            rank_zero_debug("HPU `auto_device_count` failed, returning default count of 8.")
            return 8

    @staticmethod
    def is_available() -> bool:
        """Returns a bool indicating if HPU is currently available."""
        try:
            return torch_hpu.is_available()
        except (AttributeError, NameError):
            return False

    @staticmethod
    def get_device_name() -> str:
        """Returns the name of the HPU device."""
        try:
            return torch_hpu.get_device_name()
        except (AttributeError, NameError):
            return ""

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "hpu",
            cls,
            description=f"{cls.__class__.__name__}",
        )
