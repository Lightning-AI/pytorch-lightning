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
from collections.abc import MutableSequence
from typing import Optional, Union

import torch

from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.fabric.utilities.types import _DEVICE


def _determine_root_gpu_device(gpus: list[_DEVICE]) -> Optional[_DEVICE]:
    """
    Args:
        gpus: Non-empty list of ints representing which GPUs to use

    Returns:
        Designated root GPU device id

    Raises:
        TypeError:
            If ``gpus`` is not a list
        AssertionError:
            If GPU list is empty
    """
    if gpus is None:
        return None

    if not isinstance(gpus, list):
        raise TypeError("GPUs should be a list")

    assert len(gpus) > 0, "GPUs should be a non-empty list"

    # set root gpu
    return gpus[0]


def _parse_gpu_ids(
    gpus: Optional[Union[int, str, list[int]]],
    include_cuda: bool = False,
    include_mps: bool = False,
) -> Optional[list[int]]:
    """Parses the GPU IDs given in the format as accepted by the :class:`~lightning.pytorch.trainer.trainer.Trainer`.

    Args:
        gpus: An int -1 or string '-1' indicate that all available GPUs should be used.
            A list of unique ints or a string containing a list of comma separated unique integers
            indicates specific GPUs to use.
            An int of 0 means that no GPUs should be used.
            Any int N > 0 indicates that GPUs [0..N) should be used.
        include_cuda: A boolean value indicating whether to include CUDA devices for GPU parsing.
        include_mps: A boolean value indicating whether to include MPS devices for GPU parsing.

    Returns:
        A list of GPUs to be used or ``None`` if no GPUs were requested

    Raises:
        MisconfigurationException:
            If no GPUs are available but the value of gpus variable indicates request for GPUs

    .. note::
        ``include_cuda`` and ``include_mps`` default to ``False`` so that you only
        have to specify which device type to use and all other devices are not disabled.

    """
    # Check that gpus param is None, Int, String or Sequence of Ints
    _check_data_type(gpus)

    # Handle the case when no GPUs are requested
    if gpus is None or (isinstance(gpus, int) and gpus == 0) or str(gpus).strip() in ("0", "[]"):
        return None

    # We know the user requested GPUs therefore if some of the
    # requested GPUs are not available an exception is thrown.
    gpus = _normalize_parse_gpu_string_input(gpus)
    gpus = _normalize_parse_gpu_input_to_list(gpus, include_cuda=include_cuda, include_mps=include_mps)
    if not gpus:
        raise MisconfigurationException("GPUs requested but none are available.")

    if (
        torch.distributed.is_available()
        and torch.distributed.is_torchelastic_launched()
        and len(gpus) != 1
        and len(_get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)) == 1
    ):
        # Omit sanity check on torchelastic because by default it shows one visible GPU per process
        return gpus

    # Check that GPUs are unique. Duplicate GPUs are not supported by the backend.
    _check_unique(gpus)

    return _sanitize_gpu_ids(gpus, include_cuda=include_cuda, include_mps=include_mps)


def _normalize_parse_gpu_string_input(s: Union[int, str, list[int]]) -> Union[int, list[int]]:
    if not isinstance(s, str):
        return s
    if s == "-1":
        return -1
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if len(x) > 0]
    return int(s.strip())


def _sanitize_gpu_ids(gpus: list[int], include_cuda: bool = False, include_mps: bool = False) -> list[int]:
    """Checks that each of the GPUs in the list is actually available. Raises a MisconfigurationException if any of the
    GPUs is not available.

    Args:
        gpus: List of ints corresponding to GPU indices

    Returns:
        Unmodified gpus variable

    Raises:
        MisconfigurationException:
            If machine has fewer available GPUs than requested.

    """
    if sum((include_cuda, include_mps)) == 0:
        raise ValueError("At least one gpu type should be specified!")
    all_available_gpus = _get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)
    for gpu in gpus:
        if gpu not in all_available_gpus:
            raise MisconfigurationException(
                f"You requested gpu: {gpus}\n But your machine only has: {all_available_gpus}"
            )
    return gpus


def _normalize_parse_gpu_input_to_list(
    gpus: Union[int, list[int], tuple[int, ...]], include_cuda: bool, include_mps: bool
) -> Optional[list[int]]:
    assert gpus is not None
    if isinstance(gpus, (MutableSequence, tuple)):
        return list(gpus)

    # must be an int
    if not gpus:  # gpus==0
        return None
    if gpus == -1:
        return _get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)

    return list(range(gpus))


def _get_all_available_gpus(include_cuda: bool = False, include_mps: bool = False) -> list[int]:
    """
    Returns:
        A list of all available GPUs
    """
    from lightning.fabric.accelerators.cuda import _get_all_visible_cuda_devices
    from lightning.fabric.accelerators.mps import _get_all_available_mps_gpus

    cuda_gpus = _get_all_visible_cuda_devices() if include_cuda else []
    mps_gpus = _get_all_available_mps_gpus() if include_mps else []
    return cuda_gpus + mps_gpus


def _check_unique(device_ids: list[int]) -> None:
    """Checks that the device_ids are unique.

    Args:
        device_ids: List of ints corresponding to GPUs indices

    Raises:
        MisconfigurationException:
            If ``device_ids`` of GPUs aren't unique

    """
    if len(device_ids) != len(set(device_ids)):
        raise MisconfigurationException("Device ID's (GPU) must be unique.")


def _check_data_type(device_ids: object) -> None:
    """Checks that the device_ids argument is one of the following: int, string, or sequence of integers.

    Args:
        device_ids: gpus/tpu_cores parameter as passed to the Trainer

    Raises:
        TypeError:
            If ``device_ids`` of GPU/TPUs aren't ``int``, ``str`` or sequence of ``int```

    """
    msg = "Device IDs (GPU/TPU) must be an int, a string, a sequence of ints, but you passed"
    if device_ids is None:
        raise TypeError(f"{msg} None")
    if isinstance(device_ids, (MutableSequence, tuple)):
        for id_ in device_ids:
            id_type = type(id_)  # because `isinstance(False, int)` -> True
            if id_type is not int:
                raise TypeError(f"{msg} a sequence of {type(id_).__name__}.")
    elif type(device_ids) not in (int, str):
        raise TypeError(f"{msg} {device_ids!r}.")
