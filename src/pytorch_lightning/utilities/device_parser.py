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
from typing import List, Optional, Union

from pytorch_lightning.utilities.exceptions import MisconfigurationException

# For backward-compatibility
# TODO: deprecate usage
from lightning_lite.utilities.device_parser import (  # noqa: F401
determine_root_gpu_device, parse_gpu_ids,
parse_tpu_cores, parse_cpu_cores, num_cuda_devices, is_cuda_available
)


def parse_hpus(devices: Optional[Union[int, str, List[int]]]) -> Optional[int]:
    """
    Parses the hpus given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer` for the `devices` flag.

    Args:
        devices: An integer that indicates the number of Gaudi devices to be used

    Returns:
        Either an integer or ``None`` if no devices were requested

    Raises:
        MisconfigurationException:
            If devices aren't of type `int` or `str`
    """
    if devices is not None and not isinstance(devices, (int, str)):
        raise MisconfigurationException("`devices` for `HPUAccelerator` must be int, string or None.")

    return int(devices) if isinstance(devices, str) else devices
