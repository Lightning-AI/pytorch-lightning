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
import logging
import os
import shutil
import subprocess
from typing import Any, Optional, Union

import torch
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.accelerators.musa import _check_musa_matmul_precision, _clear_musa_memory, num_musa_devices
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)


class MUSAAccelerator(Accelerator):
    """Accelerator for MUSA devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not GPU.
        """
        if device.type != "musa":
            raise MisconfigurationException(f"Device should be GPU, got {device} instead")
        _check_musa_matmul_precision(device)
        torch.musa.set_device(device)

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO refactor input from trainer to local_rank @four4fish
        self.set_musa_flags(trainer.local_rank)
        _clear_musa_memory()

    @staticmethod
    def set_musa_flags(local_rank: int) -> None:
        # set the correct musa visible devices (using pci order)
        os.environ["MUSA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join(str(x) for x in range(num_musa_devices()))
        devices = os.getenv("MUSA_VISIBLE_DEVICES", all_gpu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - MUSA_VISIBLE_DEVICES: [{devices}]")

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Gets stats for the given GPU device.

        Args:
            device: GPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.

        Raises:
            FileNotFoundError:
                If mthreds-gmi installation not found

        """
        return torch.musa.memory_stats(device)

    @override
    def teardown(self) -> None:
        _clear_musa_memory()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        """Accelerator device parsing logic."""
        return _parse_gpu_ids(devices, include_musa=True)

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("musa", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_musa_devices()

    @staticmethod
    @override
    def is_available() -> bool:
        return num_musa_devices() > 0

    @staticmethod
    @override
    def name() -> str:
        return "musa"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )
