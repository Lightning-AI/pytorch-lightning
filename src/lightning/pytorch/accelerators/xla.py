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
from typing import Any

from typing_extensions import override

from lightning.fabric.accelerators import _AcceleratorRegistry
from lightning.fabric.accelerators.xla import XLAAccelerator as FabricXLAAccelerator
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator


class XLAAccelerator(Accelerator, FabricXLAAccelerator):
    """Accelerator for XLA devices, normally TPUs.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Gets stats for the given XLA device.

        Args:
            device: XLA device for which to get stats

        Returns:
            A dictionary mapping the metrics (free memory and peak memory) to their values.

        """
        import torch_xla.core.xla_model as xm

        memory_info = xm.get_memory_info(device)
        free_memory = memory_info["kb_free"]
        peak_memory = memory_info["kb_total"] - free_memory
        return {
            "avg. free memory (MB)": free_memory,
            "avg. peak memory (MB)": peak_memory,
        }

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register("tpu", cls, description=cls.__name__)
