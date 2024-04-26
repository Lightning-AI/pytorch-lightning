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
import operator
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from lightning_utilities.core.imports import compare_version
from packaging.version import Version

from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.fabric.accelerators.mps import MPSAccelerator
from lightning.fabric.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1


def _runif_reasons(
    *,
    min_cuda_gpus: int = 0,
    min_torch: Optional[str] = None,
    max_torch: Optional[str] = None,
    min_python: Optional[str] = None,
    bf16_cuda: bool = False,
    tpu: bool = False,
    mps: Optional[bool] = None,
    skip_windows: bool = False,
    standalone: bool = False,
    deepspeed: bool = False,
    dynamo: bool = False,
) -> Tuple[List[str], Dict[str, bool]]:
    """Construct reasons for pytest skipif.

    Args:
        min_cuda_gpus: Require this number of gpus and that the ``PL_RUN_CUDA_TESTS=1`` environment variable is set.
        min_torch: Require that PyTorch is greater or equal than this version.
        max_torch: Require that PyTorch is less than this version.
        min_python: Require that Python is greater or equal than this version.
        bf16_cuda: Require that CUDA device supports bf16.
        tpu: Require that TPU is available.
        mps: If True: Require that MPS (Apple Silicon) is available,
            if False: Explicitly Require that MPS is not available
        skip_windows: Skip for Windows platform.
        standalone: Mark the test as standalone, our CI will run it in a separate process.
            This requires that the ``PL_RUN_STANDALONE_TESTS=1`` environment variable is set.
        deepspeed: Require that microsoft/DeepSpeed is installed.
        dynamo: Require that `torch.dynamo` is supported.

    """
    reasons = []
    kwargs = {}  # used in conftest.py::pytest_collection_modifyitems

    if min_cuda_gpus:
        if num_cuda_devices() < min_cuda_gpus:
            reasons.append(f"GPUs>={min_cuda_gpus}")
        kwargs["min_cuda_gpus"] = True

    # set use_base_version for nightly support
    if min_torch and compare_version("torch", operator.lt, min_torch, use_base_version=True):
        reasons.append(f"torch>={min_torch}, {torch.__version__} installed")

    # set use_base_version for nightly support
    if max_torch and compare_version("torch", operator.ge, max_torch, use_base_version=True):
        reasons.append(f"torch<{max_torch}, {torch.__version__} installed")

    if min_python:
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if Version(py_version) < Version(min_python):
            reasons.append(f"python>={min_python}")

    if bf16_cuda:
        try:
            cond = not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        except (AssertionError, RuntimeError) as ex:
            # AssertionError: Torch not compiled with CUDA enabled
            # RuntimeError: Found no NVIDIA driver on your system.
            is_unrelated = "Found no NVIDIA driver" not in str(ex) or "Torch not compiled with CUDA" not in str(ex)
            if is_unrelated:
                raise ex
            cond = True
        if cond:
            reasons.append("CUDA device bf16")

    if skip_windows and sys.platform == "win32":
        reasons.append("unimplemented on Windows")

    if tpu:
        if not XLAAccelerator.is_available():
            reasons.append("TPU")
        kwargs["tpu"] = True

    if mps is not None:
        if mps and not MPSAccelerator.is_available():
            reasons.append("MPS")
        elif not mps and MPSAccelerator.is_available():
            reasons.append("not MPS")

    if standalone:
        if os.getenv("PL_RUN_STANDALONE_TESTS", "0") != "1":
            reasons.append("Standalone execution")
        kwargs["standalone"] = True

    if deepspeed and not _DEEPSPEED_AVAILABLE:
        reasons.append("Deepspeed")

    if dynamo:
        if _TORCH_GREATER_EQUAL_2_1:
            from torch._dynamo.eval_frame import is_dynamo_supported

            cond = not is_dynamo_supported()
        else:
            cond = sys.platform == "win32" or sys.version_info >= (3, 11)
        cond |= not _TORCH_GREATER_EQUAL_2_0
        if cond:
            reasons.append("torch.dynamo")

    return reasons, kwargs
