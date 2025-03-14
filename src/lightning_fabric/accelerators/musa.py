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
from typing import cast, Dict, Generator, List, Optional, Union

import torch
from lightning_utilities.core.rank_zero import rank_zero_info

from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12, _TORCH_GREATER_EQUAL_2_0


class MUSAAccelerator(Accelerator):
    """Accelerator for NVIDIA MUSA devices."""

    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not of type MUSA.
        """
        if device.type != "musa":
            raise ValueError(f"Device should be MUSA, got {device} instead.")
        _check_musa_matmul_precision(device)
        torch.musa.set_device(device)

    def teardown(self) -> None:
        _clear_musa_memory()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        """Accelerator device parsing logic."""
        from lightning_fabric.utilities.device_parser import _parse_gpu_ids

        return _parse_gpu_ids(devices, include_musa=True)

    @staticmethod
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("musa", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        return num_musa_devices()

    @staticmethod
    def is_available() -> bool:
        return num_musa_devices() > 0

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "musa",
            cls,
            description=cls.__class__.__name__,
        )


def find_usable_musa_devices(num_devices: int = -1) -> List[int]:
    """Returns a list of all available and usable MUSA GPU devices.

    A GPU is considered usable if we can successfully move a tensor to the device, and this is what this function
    tests for each GPU on the system until the target number of usable devices is found.

    A subset of GPUs on the system might be used by other processes, and if the GPU is configured to operate in
    'exclusive' mode (configurable by the admin), then only one process is allowed to occupy it.

    Args:
        num_devices: The number of devices you want to request. By default, this function will return as many as there
            are usable MUSA GPU devices available.

    Warning:
        If multiple processes call this function at the same time, there can be race conditions in the case where
        both processes determine that the device is unoccupied, leading into one of them crashing later on.
    """
    visible_devices = _get_all_visible_musa_devices()
    if not visible_devices:
        raise ValueError(
            f"You requested to find {num_devices} devices but there are no visible MUSA devices on this machine."
        )
    if num_devices > len(visible_devices):
        raise ValueError(
            f"You requested to find {num_devices} devices but this machine only has {len(visible_devices)} GPUs."
        )

    available_devices = []
    unavailable_devices = []

    for gpu_idx in visible_devices:
        try:
            torch.tensor(0, device=torch.device("musa", gpu_idx))
        except RuntimeError:
            unavailable_devices.append(gpu_idx)
            continue

        available_devices.append(gpu_idx)
        if len(available_devices) == num_devices:
            # exit early if we found the right number of GPUs
            break

    if num_devices != -1 and len(available_devices) != num_devices:
        raise RuntimeError(
            f"You requested to find {num_devices} devices but only {len(available_devices)} are currently available."
            f" The devices {unavailable_devices} are occupied by other processes and can't be used at the moment."
        )
    return available_devices


def _get_all_visible_musa_devices() -> List[int]:
    """Returns a list of all visible MUSA GPU devices.

    Devices masked by the environment variabale ``MUSA_VISIBLE_DEVICES`` won't be returned here. For example, assume you
    have 8 physical GPUs. If ``MUSA_VISIBLE_DEVICES="1,3,6"``, then this function will return the list ``[0, 1, 2]``
    because these are the three visible GPUs after applying the mask ``MUSA_VISIBLE_DEVICES``.
    """
    return list(range(num_musa_devices()))


# TODO: Remove once minimum supported PyTorch version is 2.0
@contextmanager
def _patch_musa_is_available() -> Generator:
    """Context manager that safely patches :func:`torch.musa.is_available` with its NVML-based version if
    possible."""
    if hasattr(torch._C, "_musa_getDeviceCount") and _device_count_nvml() >= 0 and not _TORCH_GREATER_EQUAL_2_0:
        # we can safely patch is_available if both torch has MUSA compiled and the NVML count is succeeding
        # otherwise, patching is_available could lead to attribute errors or infinite recursion
        orig_check = torch.musa.is_available
        torch.musa.is_available = is_musa_available
        try:
            yield
        finally:
            torch.musa.is_available = orig_check
    else:
        yield


@lru_cache(1)
def num_musa_devices() -> int:
    """Returns the number of available MUSA devices.

    Unlike :func:`torch.musa.device_count`, this function does its best not to create a MUSA context for fork support,
    if the platform allows it.
    """
    if _TORCH_GREATER_EQUAL_2_0:
        return torch.musa.device_count()

    # Implementation copied from upstream: https://github.com/pytorch/pytorch/pull/84879
    # TODO: Remove once minimum supported PyTorch version is 2.0
    nvml_count = _device_count_nvml()
    return torch.musa.device_count() if nvml_count < 0 else nvml_count


def is_musa_available() -> bool:
    """Returns a bool indicating if MUSA is currently available.

    Unlike :func:`torch.musa.is_available`, this function does its best not to create a MUSA context for fork support,
    if the platform allows it.
    """
    # We set `PYTORCH_NVML_BASED_MUSA_CHECK=1` in lightning_fabric.__init__.py
    return torch.musa.is_available() if _TORCH_GREATER_EQUAL_2_0 else num_musa_devices() > 0


# TODO: Remove once minimum supported PyTorch version is 2.0
def _parse_visible_devices() -> Union[List[int], List[str]]:
    """Parse MUSA_VISIBLE_DEVICES environment variable."""
    var = os.getenv("MUSA_VISIBLE_DEVICES")
    if var is None:
        return list(range(64))

    def _strtoul(s: str) -> int:
        """Return -1 or positive integer sequence string starts with,"""
        if not s:
            return -1
        for idx, c in enumerate(s):
            if not (c.isdigit() or (idx == 0 and c in "+-")):
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    def parse_list_with_prefix(lst: str, prefix: str) -> List[str]:
        rcs: List[str] = []
        for elem in lst.split(","):
            # Repeated id results in empty set
            if elem in rcs:
                return cast(List[str], [])
            # Anything other but prefix is ignored
            if not elem.startswith(prefix):
                break
            rcs.append(elem)
        return rcs

    if var.startswith("GPU-"):
        return parse_list_with_prefix(var, "GPU-")
    if var.startswith("MIG-"):
        return parse_list_with_prefix(var, "MIG-")
    # MUSA_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: List[int] = []
    for elem in var.split(","):
        x = _strtoul(elem.strip())
        # Repeated ordinal results in empty set
        if x in rc:
            return cast(List[int], [])
        # Negative value aborts the sequence
        if x < 0:
            break
        rc.append(x)
    return rc


# TODO: Remove once minimum supported PyTorch version is 2.0
def _raw_device_count_nvml() -> int:
    """Return number of devices as reported by NVML or negative value if NVML discovery/initialization failed."""
    from ctypes import byref, c_int, CDLL

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML")
        return -1
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count")
        return -1
    del nvml_h
    return dev_count.value


# TODO: Remove once minimum supported PyTorch version is 2.0
def _raw_device_uuid_nvml() -> Optional[List[str]]:
    """Return list of device UUID as reported by NVML or None if NVM discovery/initialization failed."""
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML")
        return None
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count")
        return None
    uuids: List[str] = []
    for idx in range(dev_count.value):
        dev_id = c_void_p()
        rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
        if rc != 0:
            warnings.warn("Can't get device handle")
            return None
        buf_len = 96
        buf = create_string_buffer(buf_len)
        rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
        if rc != 0:
            warnings.warn("Can't get device UUID")
            return None
        uuids.append(buf.raw.decode("ascii").strip("\0"))
    del nvml_h
    return uuids


# TODO: Remove once minimum supported PyTorch version is 2.0
def _transform_uuid_to_ordinals(candidates: List[str], uuids: List[str]) -> List[int]:
    """Given the set of partial uuids and list of known uuids builds a set of ordinals excluding ambiguous partials
    IDs."""

    def uuid_to_orinal(candidate: str, uuids: List[str]) -> int:
        best_match = -1
        for idx, uuid in enumerate(uuids):
            if not uuid.startswith(candidate):
                continue
            # Ambigous candidate
            if best_match != -1:
                return -1
            best_match = idx
        return best_match

    rc: List[int] = []
    for candidate in candidates:
        idx = uuid_to_orinal(candidate, uuids)
        # First invalid ordinal stops parsing
        if idx < 0:
            break
        # Duplicates result in empty set
        if idx in rc:
            return cast(List[int], [])
        rc.append(idx)
    return rc


# TODO: Remove once minimum supported PyTorch version is 2.0
def _device_count_nvml() -> int:
    """Return number of devices as reported by NVML taking MUSA_VISIBLE_DEVICES into account.

    Negative value is returned if NVML discovery or initialization has failed.
    """
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            # Skip MIG parsing
            if visible_devices[0].startswith("MIG-"):
                return -1
            uuids = _raw_device_uuid_nvml()
            if uuids is None:
                return -1
            visible_devices = _transform_uuid_to_ordinals(cast(List[str], visible_devices), uuids)
        else:
            raw_cnt = _raw_device_count_nvml()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            for idx, val in enumerate(visible_devices):
                if cast(int, val) >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


def _check_musa_matmul_precision(device: torch.device) -> None:
    if not _TORCH_GREATER_EQUAL_1_12:
        # before 1.12, tf32 was used by default
        return
    major, _ = torch.musa.get_device_capability(device)
    ampere_or_later = major >= 8  # Ampere and later leverage tensor cores, where this setting becomes useful
    if not ampere_or_later:
        return
    # check that the user hasn't changed the precision already, this works for both `allow_tf32 = True` and
    # `set_float32_matmul_precision`
    if torch.get_float32_matmul_precision() == "highest":  # default
        rank_zero_info(
            f"You are using a MUSA device ({torch.musa.get_device_name(device)!r}) that has Tensor Cores. To properly"
            " utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off"
            " precision for performance. For more details, read https://pytorch.org/docs/stable/generated/"
            "torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"
        )
    # note: no need change `torch.backends.cudnn.allow_tf32` as it's enabled by default:
    # https://pytorch.org/docs/stable/notes/musa.html#tensorfloat-32-tf32-on-ampere-devices


def _clear_musa_memory() -> None:
    if _TORCH_GREATER_EQUAL_2_0:
        # https://github.com/pytorch/pytorch/issues/95668
        torch._C._musa_clearCublasWorkspaces()
    torch.musa.empty_cache()
