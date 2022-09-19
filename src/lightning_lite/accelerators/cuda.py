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
import multiprocessing
from typing import Dict, List, Optional, Union

import torch

from lightning_lite.accelerators.accelerator import Accelerator
from lightning_lite.strategies.launchers.multiprocessing import _is_forking_disabled


class CUDAAccelerator(Accelerator):
    """Accelerator for NVIDIA CUDA devices."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not of type CUDA.
        """
        if device.type != "cuda":
            raise ValueError(f"Device should be CUDA, got {device} instead.")
        torch.cuda.set_device(device)

    def teardown(self) -> None:
        # clean up memory
        torch.cuda.empty_cache()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        from lightning_lite.utilities.device_parser import parse_gpu_ids

        return parse_gpu_ids(devices, include_cuda=True)

    @staticmethod
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("cuda", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_cuda_devices()

    @staticmethod
    def is_available() -> bool:
        return num_cuda_devices() > 0

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "cuda",
            cls,
            description=cls.__class__.__name__,
        )


def _get_all_available_cuda_gpus() -> List[int]:
    """
    Returns:
         A list of all available CUDA GPUs
    """
    return list(range(num_cuda_devices()))


def num_cuda_devices() -> int:
    """Returns the number of GPUs available.

    Unlike :func:`torch.cuda.device_count`, this function does its best not to create a CUDA context for fork support,
    if the platform allows it.
    """
    if "fork" not in torch.multiprocessing.get_all_start_methods() or _is_forking_disabled():
        return torch.cuda.device_count()
    with multiprocessing.get_context("fork").Pool(1) as pool:
        return pool.apply(torch.cuda.device_count)


def is_cuda_available() -> bool:
    """Returns a bool indicating if CUDA is currently available.

    Unlike :func:`torch.cuda.is_available`, this function does its best not to create a CUDA context for fork support,
    if the platform allows it.
    """
    if "fork" not in torch.multiprocessing.get_all_start_methods() or _is_forking_disabled():
        return torch.cuda.is_available()
    with multiprocessing.get_context("fork").Pool(1) as pool:
        return pool.apply(torch.cuda.is_available)
