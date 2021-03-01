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
from distutils.version import LooseVersion
from typing import Optional

import pytest
import torch
from pkg_resources import get_distribution

from pytorch_lightning.utilities import _TORCH_QUANTIZE_AVAILABLE


def skipif_args(
    min_gpus: int = 0,
    min_torch: Optional[str] = None,
    quant_available: bool = False,
) -> dict:
    """ Creating aggregated arguments for standard pytest skipif, sot the usecase is::

        @pytest.mark.skipif(**create_skipif(min_torch="99"))
        def test_any_func(...):
            ...

    >>> from pprint import pprint
    >>> pprint(skipif_args(min_torch="99", min_gpus=0))
    {'condition': True, 'reason': 'Required: [torch>=99]'}
    >>> pprint(skipif_args(min_torch="0.0", min_gpus=0))  # doctest: +NORMALIZE_WHITESPACE
    {'condition': False, 'reason': 'Conditions satisfied, going ahead with the test.'}
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

    if quant_available:
        _miss_default = 'fbgemm' not in torch.backends.quantized.supported_engines
        conditions.append(not _TORCH_QUANTIZE_AVAILABLE or _miss_default)
        reasons.append("PyTorch quantization is available")

    if not any(conditions):
        return dict(condition=False, reason="Conditions satisfied, going ahead with the test.")

    reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
    return dict(condition=any(conditions), reason=f"Required: [{' + '.join(reasons)}]",)


@pytest.mark.skipif(**skipif_args(min_torch="99"))
def test_always_skip():
    exit(1)


@pytest.mark.skipif(**skipif_args(min_torch="0.0"))
def test_always_pass():
    assert True
