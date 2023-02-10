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

from typing import Any, Dict, List, Optional, Union

import torch
from lightning_utilities.core.imports import package_available

from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug

_HABANA_FRAMEWORK_AVAILABLE = package_available("habana_frameworks")

if _HABANA_FRAMEWORK_AVAILABLE:
    from habana_frameworks.torch.utils.library_loader import is_habana_available

    _HPU_AVAILABLE = is_habana_available()
else:
    _HPU_AVAILABLE = False

if _HPU_AVAILABLE:
    import habana_frameworks.torch.hpu as torch_hpu


class HPUAccelerator(Accelerator):
    """Accelerator for HPU devices."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not HPU.
        """
        if device.type != "hpu":
            raise MisconfigurationException(f"Device should be HPU, got {device} instead.")

    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
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

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[int]:
        """Accelerator device parsing logic."""
        return _parse_hpus(devices)

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
            description=cls.__class__.__name__,
        )


def _parse_hpus(devices: Optional[Union[int, str, List[int]]]) -> Optional[int]:
    """
    Parses the hpus given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer` for the `devices` flag.

    Args:
        devices: An integer that indicates the number of Gaudi devices to be used

    Returns:
        Either an integer or ``None`` if no devices were requested

    Raises:
        MisconfigurationException:
            If devices aren't of type `int` or `str`
    """
    if devices is not None and not isinstance(devices, (int, str)):
        raise MisconfigurationException("`devices` for `HPUAccelerator` must be int, string or None.")

    return int(devices) if isinstance(devices, str) else devices
