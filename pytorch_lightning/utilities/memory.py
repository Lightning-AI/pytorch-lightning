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
"""Utilities related to memory."""

import gc
import os
import shutil
import subprocess
from io import BytesIO
from typing import Any, Dict

import torch
from torch.nn import Module

from pytorch_lightning.utilities.apply_func import apply_to_collection


def recursive_detach(in_dict: Any, to_cpu: bool = False) -> Any:
    """Detach all tensors in `in_dict`.

    May operate recursively if some of the values in `in_dict` are dictionaries
    which contain instances of `torch.Tensor`. Other types in `in_dict` are
    not affected by this utility function.

    Args:
        in_dict: Dictionary with tensors to detach
        to_cpu: Whether to move tensor to cpu

    Return:
        out_dict: Dictionary with detached tensors
    """

    def detach_and_move(t: torch.Tensor, to_cpu: bool) -> torch.Tensor:
        t = t.detach()
        if to_cpu:
            t = t.cpu()
        return t

    return apply_to_collection(in_dict, torch.Tensor, detach_and_move, to_cpu=to_cpu)


def is_oom_error(exception: BaseException) -> bool:
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        # This is the last thing that should cause an OOM error, but seemingly it can.
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        if not is_oom_error(exception):
            # Only handle OOM errors
            raise


def get_memory_profile(mode: str) -> Dict[str, float]:
    r"""
    .. deprecated:: v1.5
        This function was deprecated in v1.5 in favor of
        `pytorch_lightning.accelerators.gpu._get_nvidia_gpu_stats` and will be removed in v1.7.

    Get a profile of the current memory usage.

    Args:
        mode: There are two modes:

            - 'all' means return memory for all gpus
            - 'min_max' means return memory for max and min

    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.
        If mode is 'min_max', the dictionary will also contain two additional keys:

        - 'min_gpu_mem': the minimum memory usage in MB
        - 'max_gpu_mem': the maximum memory usage in MB
    """
    memory_map = get_gpu_memory_map()

    if mode == "min_max":
        min_index, min_memory = min(memory_map.items(), key=lambda item: item[1])
        max_index, max_memory = max(memory_map.items(), key=lambda item: item[1])

        memory_map = {"min_gpu_mem": min_memory, "max_gpu_mem": max_memory}

    return memory_map


def get_gpu_memory_map() -> Dict[str, float]:
    r"""
    .. deprecated:: v1.5
        This function was deprecated in v1.5 in favor of
        `pytorch_lightning.accelerators.gpu._get_nvidia_gpu_stats` and will be removed in v1.7.

    Get the current gpu usage.

    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.

    Raises:
        FileNotFoundError:
            If nvidia-smi installation not found
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        raise FileNotFoundError("nvidia-smi: command not found")
    result = subprocess.run(
        [nvidia_smi_path, "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
        capture_output=True,
        check=True,
    )

    # Convert lines into a dictionary
    gpu_memory = [float(x) for x in result.stdout.strip().split(os.linesep)]
    gpu_memory_map = {f"gpu_id: {gpu_id}/memory.used (MB)": memory for gpu_id, memory in enumerate(gpu_memory)}
    return gpu_memory_map


def get_model_size_mb(model: Module) -> float:
    """Calculates the size of a Module in megabytes.

    The computation includes everything in the :meth:`~torch.nn.Module.state_dict`,
    i.e., by default the parameters and buffers.

    Returns:
        Number of megabytes in the parameters of the input module.
    """
    model_size = BytesIO()
    torch.save(model.state_dict(), model_size)
    size_mb = model_size.getbuffer().nbytes / 1e6
    return size_mb
