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
import platform
from functools import lru_cache
from typing import Dict, List, Optional, Union

import torch

from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12


class MPSAccelerator(Accelerator):
    """Accelerator for Metal Apple Silicon GPU devices."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not MPS.
        """
        if device.type != "mps":
            raise ValueError(f"Device should be MPS, got {device} instead.")

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        from lightning_fabric.utilities.device_parser import _parse_gpu_ids

        parsed_devices = _parse_gpu_ids(devices, include_mps=True)
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
    @lru_cache(1)
    def is_available() -> bool:
        """MPS is only available for certain torch builds starting at torch>=1.12, and is only enabled on a machine
        with the ARM-based Apple Silicon processors."""
        return (
            _TORCH_GREATER_EQUAL_1_12 and torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64")
        )

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "mps",
            cls,
            description=cls.__class__.__name__,
        )


def _get_all_available_mps_gpus() -> List[int]:
    """
    Returns:
        A list of all available MPS GPUs
    """
    return [0] if MPSAccelerator.is_available() else []
