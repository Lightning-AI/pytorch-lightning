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
import sys
from typing import Optional

import pytest
import torch
from packaging.version import Version
from pkg_resources import get_distribution

from pytorch_lightning.utilities import (
    _APEX_AVAILABLE,
    _DEEPSPEED_AVAILABLE,
    _FAIRSCALE_AVAILABLE,
    _FAIRSCALE_FULLY_SHARDED_AVAILABLE,
    _FAIRSCALE_PIPE_AVAILABLE,
    _HOROVOD_AVAILABLE,
    _NATIVE_AMP_AVAILABLE,
    _RPC_AVAILABLE,
    _TORCH_QUANTIZE_AVAILABLE,
    _TPU_AVAILABLE,
)

try:
    from horovod.common.util import nccl_built
    nccl_built()
except (ImportError, ModuleNotFoundError, AttributeError):
    _HOROVOD_NCCL_AVAILABLE = False
finally:
    _HOROVOD_NCCL_AVAILABLE = True


class RunIf:
    """
    RunIf wrapper for simple marking specific cases, fully compatible with pytest.mark::

        @RunIf(min_torch="0.0")
        @pytest.mark.parametrize("arg1", [1, 2.0])
        def test_wrapper(arg1):
            assert arg1 > 0.0
    """

    def __new__(
        self,
        *args,
        min_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        quantization: bool = False,
        amp_apex: bool = False,
        amp_native: bool = False,
        tpu: bool = False,
        horovod: bool = False,
        horovod_nccl: bool = False,
        skip_windows: bool = False,
        special: bool = False,
        rpc: bool = False,
        fairscale: bool = False,
        fairscale_pipe: bool = False,
        fairscale_fully_sharded: bool = False,
        deepspeed: bool = False,
        **kwargs
    ):
        """
        Args:
            args: native pytest.mark.skipif arguments
            min_gpus: min number of gpus required to run test
            min_torch: minimum pytorch version to run test
            max_torch: maximum pytorch version to run test
            min_python: minimum python version required to run test
            quantization: if `torch.quantization` package is required to run test
            amp_apex: NVIDIA Apex is installed
            amp_native: if native PyTorch native AMP is supported
            tpu: if TPU is available
            horovod: if Horovod is installed
            horovod_nccl: if Horovod is installed with NCCL support
            skip_windows: skip test for Windows platform (typically fo some limited torch functionality)
            special: running in special mode, outside pytest suit
            rpc: requires Remote Procedure Call (RPC)
            fairscale: if `fairscale` module is required to run the test
            fairscale_pipe: if `fairscale` with pipe module is required to run the test
            fairscale_fully_sharded: if `fairscale` fully sharded module is required to run the test
            deepspeed: if `deepspeed` module is required to run the test
            kwargs: native pytest.mark.skipif keyword arguments
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if quantization:
            _miss_default = 'fbgemm' not in torch.backends.quantized.supported_engines
            conditions.append(not _TORCH_QUANTIZE_AVAILABLE or _miss_default)
            reasons.append("PyTorch quantization")

        if amp_native:
            conditions.append(not _NATIVE_AMP_AVAILABLE)
            reasons.append("native AMP")

        if amp_apex:
            conditions.append(not _APEX_AVAILABLE)
            reasons.append("NVIDIA Apex")

        if skip_windows:
            conditions.append(sys.platform == "win32")
            reasons.append("unimplemented on Windows")

        if tpu:
            conditions.append(not _TPU_AVAILABLE)
            reasons.append("TPU")

        if horovod:
            conditions.append(not _HOROVOD_AVAILABLE)
            reasons.append("Horovod")

        if horovod_nccl:
            conditions.append(not _HOROVOD_NCCL_AVAILABLE)
            reasons.append("Horovod with NCCL")

        if special:
            env_flag = os.getenv("PL_RUNNING_SPECIAL_TESTS", '0')
            conditions.append(env_flag != '1')
            reasons.append("Special execution")

        if rpc:
            conditions.append(not _RPC_AVAILABLE)
            reasons.append("RPC")

        if fairscale:
            conditions.append(not _FAIRSCALE_AVAILABLE)
            reasons.append("Fairscale")

        if fairscale_pipe:
            conditions.append(not _FAIRSCALE_PIPE_AVAILABLE)
            reasons.append("Fairscale Pipe")

        if fairscale_fully_sharded:
            conditions.append(not _FAIRSCALE_FULLY_SHARDED_AVAILABLE)
            reasons.append("Fairscale Fully Sharded")

        if deepspeed:
            conditions.append(not _DEEPSPEED_AVAILABLE)
            reasons.append("Deepspeed")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            *args,
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )


@RunIf(min_torch="99")
def test_always_skip():
    exit(1)


@pytest.mark.parametrize("arg1", [0.5, 1.0, 2.0])
@RunIf(min_torch="0.0")
def test_wrapper(arg1: float):
    assert arg1 > 0.0
