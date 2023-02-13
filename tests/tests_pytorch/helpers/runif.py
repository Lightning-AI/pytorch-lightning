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
from typing import Optional

import pytest
import torch
from lightning_utilities.core.imports import compare_version
from packaging.version import Version

from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.pytorch.accelerators.hpu import _HPU_AVAILABLE
from lightning.pytorch.accelerators.ipu import _IPU_AVAILABLE
from lightning.pytorch.accelerators.mps import MPSAccelerator
from lightning.pytorch.accelerators.tpu import TPUAccelerator
from lightning.pytorch.callbacks.progress.rich_progress import _RICH_AVAILABLE
from lightning.pytorch.strategies.bagua import _BAGUA_AVAILABLE
from lightning.pytorch.strategies.colossalai import _COLOSSALAI_AVAILABLE
from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning.pytorch.utilities.imports import _OMEGACONF_AVAILABLE, _PSUTIL_AVAILABLE, _TORCH_QUANTIZE_AVAILABLE
from tests_pytorch.helpers.datamodules import _SKLEARN_AVAILABLE


class RunIf:
    """RunIf wrapper for simple marking specific cases, fully compatible with pytest.mark::

    @RunIf(min_torch="0.0")
    @pytest.mark.parametrize("arg1", [1, 2.0])
    def test_wrapper(arg1):
        assert arg1 > 0.0
    """

    def __new__(
        self,
        *args,
        min_cuda_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        quantization: bool = False,
        bf16_cuda: bool = False,
        tpu: bool = False,
        ipu: bool = False,
        hpu: bool = False,
        mps: Optional[bool] = None,
        skip_windows: bool = False,
        standalone: bool = False,
        deepspeed: bool = False,
        rich: bool = False,
        omegaconf: bool = False,
        bagua: bool = False,
        colossalai: bool = False,
        psutil: bool = False,
        sklearn: bool = False,
        **kwargs,
    ):
        """
        Args:
            *args: Any :class:`pytest.mark.skipif` arguments.
            min_cuda_gpus: Require this number of gpus and that the ``PL_RUN_CUDA_TESTS=1`` environment variable is set.
            min_torch: Require that PyTorch is greater or equal than this version.
            max_torch: Require that PyTorch is less than this version.
            min_python: Require that Python is greater or equal than this version.
            quantization: Require that `torch.quantization` is available.
            bf16_cuda: Require that CUDA device supports bf16.
            tpu: Require that TPU is available.
            ipu: Require that IPU is available and that the ``PL_RUN_IPU_TESTS=1`` environment variable is set.
            hpu: Require that HPU is available.
            mps: If True: Require that MPS (Apple Silicon) is available,
                if False: Explicitly Require that MPS is not available
            skip_windows: Skip for Windows platform.
            standalone: Mark the test as standalone, our CI will run it in a separate process.
                This requires that the ``PL_RUN_STANDALONE_TESTS=1`` environment variable is set.
            deepspeed: Require that microsoft/DeepSpeed is installed.
            rich: Require that willmcgugan/rich is installed.
            omegaconf: Require that omry/omegaconf is installed.
            bagua: Require that BaguaSys/bagua is installed.
            psutil: Require that psutil is installed.
            sklearn: Require that scikit-learn is installed.
            **kwargs: Any :class:`pytest.mark.skipif` keyword arguments.
        """
        conditions = []
        reasons = []

        if min_cuda_gpus:
            conditions.append(num_cuda_devices() < min_cuda_gpus)
            reasons.append(f"GPUs>={min_cuda_gpus}")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["min_cuda_gpus"] = True

        if min_torch:
            # set use_base_version for nightly support
            conditions.append(compare_version("torch", operator.lt, min_torch, use_base_version=True))
            reasons.append(f"torch>={min_torch}, {torch.__version__} installed")

        if max_torch:
            # set use_base_version for nightly support
            conditions.append(compare_version("torch", operator.ge, max_torch, use_base_version=True))
            reasons.append(f"torch<{max_torch}, {torch.__version__} installed")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if quantization:
            _miss_default = "fbgemm" not in torch.backends.quantized.supported_engines
            conditions.append(not _TORCH_QUANTIZE_AVAILABLE or _miss_default)
            reasons.append("PyTorch quantization")

        if bf16_cuda:
            try:
                cond = not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            except (AssertionError, RuntimeError) as e:
                # AssertionError: Torch not compiled with CUDA enabled
                # RuntimeError: Found no NVIDIA driver on your system.
                is_unrelated = "Found no NVIDIA driver" not in str(e) or "Torch not compiled with CUDA" not in str(e)
                if is_unrelated:
                    raise e
                cond = True

            conditions.append(cond)
            reasons.append("CUDA device bf16")

        if skip_windows:
            conditions.append(sys.platform == "win32")
            reasons.append("unimplemented on Windows")

        if tpu:
            conditions.append(not TPUAccelerator.is_available())
            reasons.append("TPU")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["tpu"] = True

        if ipu:
            conditions.append(not _IPU_AVAILABLE)
            reasons.append("IPU")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["ipu"] = True

        if hpu:
            conditions.append(not _HPU_AVAILABLE)
            reasons.append("HPU")

        if mps is not None:
            if mps:
                conditions.append(not MPSAccelerator.is_available())
                reasons.append("MPS")
            else:
                conditions.append(MPSAccelerator.is_available())
                reasons.append("not MPS")

        if standalone:
            env_flag = os.getenv("PL_RUN_STANDALONE_TESTS", "0")
            conditions.append(env_flag != "1")
            reasons.append("Standalone execution")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["standalone"] = True

        if deepspeed:
            conditions.append(not _DEEPSPEED_AVAILABLE)
            reasons.append("Deepspeed")

        if rich:
            conditions.append(not _RICH_AVAILABLE)
            reasons.append("Rich")

        if omegaconf:
            conditions.append(not _OMEGACONF_AVAILABLE)
            reasons.append("omegaconf")

        if bagua:
            conditions.append(not _BAGUA_AVAILABLE or sys.platform in ("win32", "darwin"))
            reasons.append("Bagua")

        if colossalai:
            conditions.append(not _COLOSSALAI_AVAILABLE)
            reasons.append("ColossalAI")

        if psutil:
            conditions.append(not _PSUTIL_AVAILABLE)
            reasons.append("psutil")

        if sklearn:
            conditions.append(not _SKLEARN_AVAILABLE)
            reasons.append("scikit-learn")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            *args, condition=any(conditions), reason=f"Requires: [{' + '.join(reasons)}]", **kwargs
        )


@RunIf(min_torch="99")
def test_always_skip():
    exit(1)


@pytest.mark.parametrize("arg1", [0.5, 1.0, 2.0])
@RunIf(min_torch="0.0")
def test_wrapper(arg1: float):
    assert arg1 > 0.0
