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
import os
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Generator, List, Optional, Set, Union

import torch

from lightning_lite.accelerators.accelerator import Accelerator
from lightning_lite.utilities.imports import _TORCH_GREATER_EQUAL_1_13, _TORCH_GREATER_EQUAL_1_14


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
        from lightning_lite.utilities.device_parser import _parse_gpu_ids

        return _parse_gpu_ids(devices, include_cuda=True)

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


# TODO: Remove once minimum supported PyTorch version is 1.14
@contextmanager
def _patch_cuda_is_available() -> Generator:
    """Context manager that safely patches :func:`torch.cuda.is_available` with its NVML-based version if
    possible."""
    if hasattr(torch._C, "_cuda_getDeviceCount") and _device_count_nvml() >= 0 and not _TORCH_GREATER_EQUAL_1_14:
        # we can safely patch is_available if both torch has CUDA compiled and the NVML count is succeeding
        # otherwise, patching is_available could lead to attribute errors or infinite recursion
        orig_check = torch.cuda.is_available
        torch.cuda.is_available = is_cuda_available
        try:
            yield
        finally:
            torch.cuda.is_available = orig_check
    else:
        yield


@lru_cache(1)
def num_cuda_devices() -> int:
    """Returns the number of available CUDA devices.

    Unlike :func:`torch.cuda.device_count`, this function does its best not to create a CUDA context for fork support,
    if the platform allows it.
    """
    if _TORCH_GREATER_EQUAL_1_13:
        return torch.cuda.device_count()

    # Implementation copied from upstream: https://github.com/pytorch/pytorch/pull/84879
    # TODO: Remove once minimum supported PyTorch version is 1.13
    nvml_count = _device_count_nvml()
    return torch.cuda.device_count() if nvml_count < 0 else nvml_count


def is_cuda_available() -> bool:
    """Returns a bool indicating if CUDA is currently available.

    Unlike :func:`torch.cuda.is_available`, this function does its best not to create a CUDA context for fork support,
    if the platform allows it.
    """
    # We set `PYTORCH_NVML_BASED_CUDA_CHECK=1` in lightning_lite.__init__.py
    return torch.cuda.is_available() if _TORCH_GREATER_EQUAL_1_14 else num_cuda_devices() > 0


# TODO: Remove once minimum supported PyTorch version is 1.13
def _parse_visible_devices() -> Set[int]:
    """Implementation copied from upstream: https://github.com/pytorch/pytorch/pull/84879."""
    var = os.getenv("CUDA_VISIBLE_DEVICES")
    if var is None:
        return {x for x in range(64)}

    def _strtoul(s: str) -> int:
        """Return -1 or integer sequence string starts with."""
        if len(s) == 0:
            return -1
        for idx, c in enumerate(s):
            if not c.isdigit():
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    # CUDA_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: Set[int] = set()
    for elem in var.split(","):
        rc.add(_strtoul(elem.strip()))
    return rc


# TODO: Remove once minimum supported PyTorch version is 1.13
def _raw_device_count_nvml() -> int:
    """Implementation copied from upstream: https://github.com/pytorch/pytorch/pull/84879."""
    from ctypes import c_int, CDLL

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML")
        return -1
    dev_arr = (c_int * 1)(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(dev_arr)
    if rc != 0:
        warnings.warn("Can't get nvml device count")
        return -1
    del nvml_h
    return dev_arr[0]


# TODO: Remove once minimum supported PyTorch version is 1.13
def _device_count_nvml() -> int:
    """Implementation copied from upstream: https://github.com/pytorch/pytorch/pull/84879."""
    try:
        raw_cnt = _raw_device_count_nvml()
        if raw_cnt <= 0:
            return raw_cnt
        return len(set(range(raw_cnt)).intersection(_parse_visible_devices()))
    except OSError:
        return -1
    except AttributeError:
        return -1
