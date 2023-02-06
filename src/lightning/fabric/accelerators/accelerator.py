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
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class Accelerator(ABC):
    """The Accelerator base class.

    An Accelerator is meant to deal with one type of hardware.
    """

    @abstractmethod
    def setup_device(self, device: torch.device) -> None:
        """Create and prepare the device for the current process."""

    @abstractmethod
    def teardown(self) -> None:
        """Clean up any state created by the accelerator."""

    @staticmethod
    @abstractmethod
    def parse_devices(devices: Any) -> Any:
        """Accelerator device parsing logic."""

    @staticmethod
    @abstractmethod
    def get_parallel_devices(devices: Any) -> Any:
        """Gets parallel devices for the Accelerator."""

    @staticmethod
    @abstractmethod
    def auto_device_count() -> int:
        """Get the device count when set to auto."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Detect if the hardware is available."""

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        pass
