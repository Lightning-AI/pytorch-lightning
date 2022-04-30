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
from pytorch_lightning.utilities import _HPU_AVAILABLE, device_parser
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug


class HPUAccelerator(Accelerator):
    """Accelerator for HPU devices."""

    def setup_environment(self, root_device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not HPU.
        """
        super().setup_environment(root_device)
        if root_device.type != "hpu":
            raise MisconfigurationException(f"Device should be HPU, got {root_device} instead.")

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        """HPU device stats aren't supported yet."""
        rank_zero_debug("HPU device stats aren't supported yet.")
        return {}

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[int]:
        """Accelerator device parsing logic."""
        return device_parser.parse_hpus(devices)

    @staticmethod
    def get_parallel_devices(devices: int) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("hpu")] * devices

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        # TODO(@kaushikb11): Update this when api is exposed by the Habana team
        return 8

    @staticmethod
    def is_available() -> bool:
        return _HPU_AVAILABLE

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "hpu",
            cls,
            description=f"{cls.__class__.__name__}",
        )
