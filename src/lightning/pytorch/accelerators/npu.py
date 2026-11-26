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
"""Accelerator for Huawei Ascend NPU devices — PyTorch Lightning side.

Mirrors ``CUDAAccelerator`` (pytorch side): it subclasses the PyTorch-Lightning
``Accelerator`` so it is picked up by ``lightning.pytorch.accelerators`` registry
auto-registration, and delegates the shared NPU device parsing to the Fabric helper.
"""

import logging
import os
from typing import Any, Optional, Union

import torch
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.accelerators.npu import _is_npu_available, _parse_npu_devices, num_npu_devices
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException

_log = logging.getLogger(__name__)


class NPUAccelerator(Accelerator):
    """Accelerator for Huawei Ascend NPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not NPU.
        """
        if device.type != "npu":
            raise MisconfigurationException(f"Device should be NPU, got {device} instead")
        torch.npu.set_device(device)

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO refactor input from trainer to local_rank @four4fish
        self.set_ascend_flags(trainer.local_rank)
        torch.npu.empty_cache()

    @staticmethod
    def set_ascend_flags(local_rank: int) -> None:
        # log the visible NPU device ids (device visibility is handled by ASCEND_RT_VISIBLE_DEVICES),
        # mirroring `CUDAAccelerator.set_nvidia_flags`
        all_npu_ids = ",".join(str(x) for x in range(num_npu_devices()))
        devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", all_npu_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - ASCEND_RT_VISIBLE_DEVICES: [{devices}]")

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Gets stats for the given NPU device (memory info reported by the NPU driver)."""
        free_bytes, total_bytes = torch.npu.mem_get_info(device)
        return {"num_free": free_bytes, "num_total": total_bytes}

    @override
    def teardown(self) -> None:
        torch.npu.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        """Accelerator device parsing logic."""
        return _parse_npu_devices(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("npu", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_npu_devices()

    @staticmethod
    @override
    def is_available() -> bool:
        return _is_npu_available()

    @staticmethod
    @override
    def name() -> str:
        return "npu"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )
