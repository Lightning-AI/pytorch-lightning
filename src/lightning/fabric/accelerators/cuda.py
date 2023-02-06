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
import os
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, Generator, List, Optional, Set, Union

import torch
from lightning_utilities.core.rank_zero import rank_zero_info

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_1_12,
    _TORCH_GREATER_EQUAL_1_13,
    _TORCH_GREATER_EQUAL_2_0,
)


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
        _check_cuda_matmul_precision(device)
        torch.cuda.set_device(device)

    def teardown(self) -> None:
        # clean up memory
        torch.cuda.empty_cache()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        from lightning.fabric.utilities.device_parser import _parse_gpu_ids

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


def find_usable_cuda_devices(num_devices: int = -1) -> List[int]:
    """Returns a list of all available and usable CUDA GPU devices.

    A GPU is considered usable if we can successfully move a tensor to the device, and this is what this function
    tests for each GPU on the system until the target number of usable devices is found.

    A subset of GPUs on the system might be used by other processes, and if the GPU is configured to operate in
    'exclusive' mode (configurable by the admin), then only one process is allowed to occupy it.

    Args:
        num_devices: The number of devices you want to request. By default, this function will return as many as there
            are usable CUDA GPU devices available.

    Warning:
        If multiple processes call this function at the same time, there can be race conditions in the case where
        both processes determine that the device is unoccupied, leading into one of them crashing later on.
    """
    visible_devices = _get_all_visible_cuda_devices()
    if not visible_devices:
        raise ValueError(
            f"You requested to find {num_devices} devices but there are no visible CUDA devices on this machine."
        )
    if num_devices > len(visible_devices):
        raise ValueError(
            f"You requested to find {num_devices} devices but this machine only has {len(visible_devices)} GPUs."
        )

    available_devices = []
    unavailable_devices = []

    for gpu_idx in visible_devices:
        try:
            torch.tensor(0, device=torch.device("cuda", gpu_idx))
        except RuntimeError:
            unavailable_devices.append(gpu_idx)
            continue

        available_devices.append(gpu_idx)
        if len(available_devices) == num_devices:
            # exit early if we found the right number of GPUs
            break

    if len(available_devices) != num_devices:
        raise RuntimeError(
            f"You requested to find {num_devices} devices but only {len(available_devices)} are currently available."
            f" The devices {unavailable_devices} are occupied by other processes and can't be used at the moment."
        )
    return available_devices


def _get_all_visible_cuda_devices() -> List[int]:
    """Returns a list of all visible CUDA GPU devices.

    Devices masked by the environment variabale ``CUDA_VISIBLE_DEVICES`` won't be returned here. For example, assume you
    have 8 physical GPUs. If ``CUDA_VISIBLE_DEVICES="1,3,6"``, then this function will return the list ``[0, 1, 2]``
    because these are the three visible GPUs after applying the mask ``CUDA_VISIBLE_DEVICES``.
    """
    return list(range(num_cuda_devices()))


# TODO: Remove once minimum supported PyTorch version is 2.0
@contextmanager
def _patch_cuda_is_available() -> Generator:
    """Context manager that safely patches :func:`torch.cuda.is_available` with its NVML-based version if
    possible."""
    if hasattr(torch._C, "_cuda_getDeviceCount") and _device_count_nvml() >= 0 and not _TORCH_GREATER_EQUAL_2_0:
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
    # We set `PYTORCH_NVML_BASED_CUDA_CHECK=1` in lightning.fabric.__init__.py
    return torch.cuda.is_available() if _TORCH_GREATER_EQUAL_2_0 else num_cuda_devices() > 0


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


def _check_cuda_matmul_precision(device: torch.device) -> None:
    if not _TORCH_GREATER_EQUAL_1_12:
        # before 1.12, tf32 was used by default
        return
    major, _ = torch.cuda.get_device_capability(device)
    ampere_or_later = major >= 8  # Ampere and later leverage tensor cores, where this setting becomes useful
    if not ampere_or_later:
        return
    # check that the user hasn't changed the precision already, this works for both `allow_tf32 = True` and
    # `set_float32_matmul_precision`
    if torch.get_float32_matmul_precision() == "highest":  # default
        rank_zero_info(
            f"You are using a CUDA device ({torch.cuda.get_device_name(device)!r}) that has Tensor Cores. To properly"
            " utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off"
            " precision for performance. For more details, read https://pytorch.org/docs/stable/generated/"
            "torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"
        )
    # note: no need change `torch.backends.cudnn.allow_tf32` as it's enabled by default:
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
