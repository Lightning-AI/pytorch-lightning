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
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.utilities.imports import _TPU_AVAILABLE, _XLA_AVAILABLE

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


class TPUAccelerator(Accelerator):
    """Accelerator for TPU devices."""

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """Gets stats for the given TPU device.

        Args:
            device: TPU device for which to get stats

        Returns:
            A dictionary mapping the metrics (free memory and peak memory) to their values.
        """
        memory_info = xm.get_memory_info(device)
        free_memory = memory_info["kb_free"]
        peak_memory = memory_info["kb_total"] - free_memory
        device_stats = {
            "avg. free memory (MB)": free_memory,
            "avg. peak memory (MB)": peak_memory,
        }
        return device_stats

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
            description=f"{cls.__class__.__name__}",
        )
