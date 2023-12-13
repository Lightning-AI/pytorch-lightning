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
from typing import Dict, List, Optional, Tuple

from lightning_utilities.core.imports import RequirementCache

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning.fabric.utilities.testing import _runif_reasons as fabric_run_if
from lightning.pytorch.accelerators.cpu import _PSUTIL_AVAILABLE
from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.core.module import _ONNX_AVAILABLE
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE

_SKLEARN_AVAILABLE = RequirementCache("scikit-learn")


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
    rich: bool = False,
    omegaconf: bool = False,
    psutil: bool = False,
    sklearn: bool = False,
    onnx: bool = False,
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
        rich: Require that willmcgugan/rich is installed.
        omegaconf: Require that omry/omegaconf is installed.
        psutil: Require that psutil is installed.
        sklearn: Require that scikit-learn is installed.
        onnx: Require that onnx is installed.

    """

    reasons, kwargs = fabric_run_if(
        min_cuda_gpus=min_cuda_gpus,
        min_torch=min_torch,
        max_torch=max_torch,
        min_python=min_python,
        bf16_cuda=bf16_cuda,
        tpu=tpu,
        mps=mps,
        skip_windows=skip_windows,
        standalone=standalone,
        deepspeed=deepspeed,
        dynamo=dynamo,
    )

    if rich and not _RICH_AVAILABLE:
        reasons.append("Rich")

    if omegaconf and not _OMEGACONF_AVAILABLE:
        reasons.append("omegaconf")

    if psutil and not _PSUTIL_AVAILABLE:
        reasons.append("psutil")

    if sklearn and not _SKLEARN_AVAILABLE:
        reasons.append("scikit-learn")

    if onnx and _TORCH_GREATER_EQUAL_2_0 and not _ONNX_AVAILABLE:
        reasons.append("onnx")

    return reasons, kwargs
