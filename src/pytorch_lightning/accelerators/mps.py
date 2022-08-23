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
import platform
from typing import Any, Dict, List, Optional, Union

import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _PSUTIL_AVAILABLE, _TORCH_GREATER_EQUAL_1_12
from pytorch_lightning.utilities.types import _DEVICE

# For using the `MPSAccelerator`, user's machine should have `torch>=1.12`, Metal programming framework and
# the ARM-based Apple Silicon processors.
_MPS_AVAILABLE = (
    _TORCH_GREATER_EQUAL_1_12 and torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64")
)


class MPSAccelerator(Accelerator):
    """Accelerator for Metal Apple Silicon GPU devices."""

    def setup_environment(self, root_device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not MPS.
        """
        super().setup_environment(root_device)
        if root_device.type != "mps":
            raise MisconfigurationException(f"Device should be MPS, got {root_device} instead.")

    def get_device_stats(self, device: _DEVICE) -> Dict[str, Any]:
        """Get M1 (cpu + gpu) stats from ``psutil`` package."""
        return get_device_stats()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        parsed_devices = device_parser.parse_gpu_ids(devices, include_mps=True)
        return parsed_devices

    @staticmethod
    def get_parallel_devices(devices: Union[int, str, List[int]]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        parsed_devices = MPSAccelerator.parse_devices(devices)
        assert parsed_devices is not None

        return [torch.device("mps", i) for i in range(len(parsed_devices))]

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1

    @staticmethod
    def is_available() -> bool:
        """MPS is only available for certain torch builds starting at torch>=1.12."""
        return _MPS_AVAILABLE

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "mps",
            cls,
            description=cls.__class__.__name__,
        )


# device metrics
_VM_PERCENT = "M1_vm_percent"
_PERCENT = "M1_percent"
_SWAP_PERCENT = "M1_swap_percent"


def get_device_stats() -> Dict[str, float]:
    if not _PSUTIL_AVAILABLE:
        raise ModuleNotFoundError(
            "Fetching M1 device stats requires `psutil` to be installed."
            " Install it by running `pip install -U psutil`."
        )
    import psutil

    return {
        _VM_PERCENT: psutil.virtual_memory().percent,
        _PERCENT: psutil.cpu_percent(),
        _SWAP_PERCENT: psutil.swap_memory().percent,
    }
