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
from typing import Any, Dict, List, Union

import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _PSUTIL_AVAILABLE
from pytorch_lightning.utilities.types import _DEVICE

if _PSUTIL_AVAILABLE:
    import psutil

# CPU device metrics
CPU_VM_PERCENT = "cpu_vm_percent"
CPU_PERCENT = "cpu_percent"
CPU_SWAP_PERCENT = "cpu_swap_percent"


class CPUAccelerator(Accelerator):
    """Accelerator for CPU devices."""

    def setup_environment(self, root_device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not CPU.
        """
        super().setup_environment(root_device)
        if root_device.type != "cpu":
            raise MisconfigurationException(
                f"Device should be CPU, got {root_device} instead."
            )

    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Get CPU stats from psutil."""
        return get_cpu_process_metrics()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> int:
        """Accelerator device parsing logic."""
        devices = device_parser.parse_cpu_cores(devices)
        return devices

    @staticmethod
    def get_parallel_devices(devices: Union[int, str, List[int]]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        devices = device_parser.parse_cpu_cores(devices)
        return [torch.device("cpu")] * devices

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 1

    @staticmethod
    def is_available() -> bool:
        """CPU is always available for execution."""
        return True

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "cpu",
            cls,
            description=f"{cls.__class__.__name__}",
        )

def get_cpu_process_metrics() -> dict[str, float]:
    if not _PSUTIL_AVAILABLE:
        raise ModuleNotFoundError(
            "`get_cpu_process_metrics` requires `psutil` to be installed. "
            + "Install it by running `pip install -U psutil`."
        )

    metrics = {
        CPU_VM_PERCENT: psutil.virtual_memory().percent,
        CPU_PERCENT: psutil.cpu_percent(),
        CPU_SWAP_PERCENT: psutil.swap_memory().percent,
    }
    return metrics
