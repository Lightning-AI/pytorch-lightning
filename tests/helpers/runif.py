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
import sys
from distutils.version import LooseVersion
from typing import Optional

import pytest
import torch
from pkg_resources import get_distribution

from pytorch_lightning.utilities import _TORCH_QUANTIZE_AVAILABLE


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
        quantization: bool = False,
        not_windows: bool = False,
        **kwargs
    ):
        """
        Args:
            args: native pytest.mark.skipif arguments
            min_gpus: min number of gpus required to run test
            min_torch: minimum pytorch version to run test
            quantization: if `torch.quantization` package is required to run test
            kwargs: native pytest.mark.skipif keyword arguments
        """
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = LooseVersion(get_distribution("torch").version)
            conditions.append(torch_version < LooseVersion(min_torch))
            reasons.append(f"torch>={min_torch}")

        if quantization:
            _miss_default = 'fbgemm' not in torch.backends.quantized.supported_engines
            conditions.append(not _TORCH_QUANTIZE_AVAILABLE or _miss_default)
            reasons.append("missing PyTorch quantization")

        if not_windows:
            conditions.append(sys.platform == "win32")
            reasons.append("unimplemented on Windows")

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
def test_wrapper(arg1):
    assert arg1 > 0.0
