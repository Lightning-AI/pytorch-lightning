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
from typing import Dict, List, Union

import torch

from lightning_fabric.accelerators.accelerator import Accelerator


class CPUAccelerator(Accelerator):
    """Accelerator for CPU devices."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not CPU.
        """
        if device.type != "cpu":
            raise ValueError(f"Device should be CPU, got {device} instead.")

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> int:
        """Accelerator device parsing logic."""
        devices = _parse_cpu_cores(devices)
        return devices

    @staticmethod
    def get_parallel_devices(devices: Union[int, str, List[int]]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        devices = _parse_cpu_cores(devices)
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
            description=cls.__class__.__name__,
        )


def _parse_cpu_cores(cpu_cores: Union[int, str, List[int]]) -> int:
    """Parses the cpu_cores given in the format as accepted by the ``devices`` argument in the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        cpu_cores: An int > 0.

    Returns:
        An int representing the number of processes

    Raises:
        MisconfigurationException:
            If cpu_cores is not an int > 0
    """
    if isinstance(cpu_cores, str) and cpu_cores.strip().isdigit():
        cpu_cores = int(cpu_cores)

    if not isinstance(cpu_cores, int) or cpu_cores <= 0:
        raise TypeError("`devices` selected with `CPUAccelerator` should be an int > 0.")

    return cpu_cores
