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

from lightning_lite.accelerators.accelerator import Accelerator
from lightning_lite.utilities.device_parser import _check_data_type
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
        return parse_tpu_cores(devices)

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


def parse_tpu_cores(tpu_cores: Optional[Union[int, str, List[int]]]) -> Optional[Union[int, List[int]]]:
    """
    Parses the tpu_cores given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        tpu_cores: An int of 1 or string '1' indicates that 1 core with multi-processing should be used
            An int 8 or string '8' indicates that all 8 cores with multi-processing should be used
            A list of ints or a strings containing a list of comma separated integers
            indicates the specific TPU core to use.

    Returns:
        A list of tpu_cores to be used or ``None`` if no TPU cores were requested

    Raises:
        MisconfigurationException:
            If TPU cores aren't 1, 8 or [<1-8>]
    """
    _check_data_type(tpu_cores)

    if isinstance(tpu_cores, str):
        tpu_cores = _parse_tpu_cores_str(tpu_cores.strip())

    if not _tpu_cores_valid(tpu_cores):
        raise TypeError("`tpu_cores` can only be 1, 8 or [<1-8>]")

    return tpu_cores


def _tpu_cores_valid(tpu_cores: Any) -> bool:
    # allow 1 or 8 cores
    if tpu_cores in (1, 8, None):
        return True

    # allow picking 1 of 8 indexes
    if isinstance(tpu_cores, (list, tuple, set)):
        has_1_tpu_idx = len(tpu_cores) == 1
        is_valid_tpu_idx = 1 <= list(tpu_cores)[0] <= 8

        is_valid_tpu_core_choice = has_1_tpu_idx and is_valid_tpu_idx
        return is_valid_tpu_core_choice

    return False


def _parse_tpu_cores_str(tpu_cores: str) -> Union[int, List[int]]:
    if tpu_cores in ("1", "8"):
        return int(tpu_cores)
    return [int(x.strip()) for x in tpu_cores.split(",") if len(x) > 0]
