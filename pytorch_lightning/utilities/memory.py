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

import gc
import os
import shutil
import subprocess
from typing import Dict, Union
import uuid

import torch
from torch.nn import Module


def recursive_detach(in_dict: dict, to_cpu: bool = False) -> dict:
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
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, dict):
            v = recursive_detach(v, to_cpu=to_cpu)
        elif callable(getattr(v, "detach", None)):
            v = v.detach()
            if to_cpu:
                v = v.cpu()
        out_dict[k] = v
    return out_dict


def is_oom_error(exception):
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception):
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def garbage_collection_cuda():
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        # This is the last thing that should cause an OOM error, but seemingly it can.
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        if not is_oom_error(exception):
            # Only handle OOM errors
            raise


def get_memory_profile(mode: str) -> Union[Dict[str, int], Dict[int, int]]:
    """Get a profile of the current memory usage.

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


def get_gpu_memory_map() -> Dict[str, int]:
    """
    Get the current gpu usage.

    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.
    """
    result = subprocess.run(
        [shutil.which("nvidia-smi"), "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
        # capture_output=True,          # valid for python version >=3.7
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
        check=True,
    )

    # Convert lines into a dictionary
    gpu_memory = [float(x) for x in result.stdout.strip().split(os.linesep)]
    gpu_memory_map = {f"gpu_id: {gpu_id}/memory.used (MB)": memory for gpu_id, memory in enumerate(gpu_memory)}
    return gpu_memory_map

  
def get_model_size_mb(model: Module) -> float:
    """
    Calculates the size of a Module in megabytes by saving the model to a temporary file and reading its size.

    The computation includes everything in the :meth:`~torch.nn.Module.state_dict`,
    i.e., by default the parameteters and buffers.

    Returns:
        Number of megabytes in the parameters of the input module.
    """
    # TODO: Implement a method without needing to download the model
    tmp_name = f"{uuid.uuid4().hex}.pt"
    torch.save(model.state_dict(), tmp_name)
    size_mb = os.path.getsize(tmp_name) / 1e6
    os.remove(tmp_name)
    return size_mb

