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
from typing import Dict, List, Optional, Union

import torch

from lightning_lite.accelerators.accelerator import Accelerator
from lightning_lite.utilities import device_parser
from lightning_lite.utilities.imports import _TPU_AVAILABLE


class TPUAccelerator(Accelerator):
    """Accelerator for TPU devices."""

    def setup_device(self, device: torch.device) -> None:
        pass

    def teardown(self) -> None:
        pass

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[Union[int, List[int]]]:
        """Accelerator device parsing logic."""
        return device_parser.parse_tpu_cores(devices)

    @staticmethod
    def get_parallel_devices(devices: Union[int, List[int]]) -> List[int]:
        """Gets parallel devices for the Accelerator."""
        if isinstance(devices, int):
            return list(range(devices))
        return devices

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return 8

    @staticmethod
    def is_available() -> bool:
        return _TPU_AVAILABLE

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "tpu",
            cls,
            description=cls.__class__.__name__,
        )
