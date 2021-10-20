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
from unittest.mock import patch, Mock

import pytest
import torch

from pytorch_lightning.lite import LightningLite
from pytorch_lightning.lite.wrappers import _LiteModule, _LiteOptimizer, _LiteDataLoader
from tests.helpers.runif import RunIf


class EmptyLite(LightningLite):
    def run(self):
        pass


@pytest.mark.parametrize(
    "precision, input_type, expected_type",
    [
        (32, torch.float32, torch.float32),
        (32, torch.float16, torch.float32),
        (32, torch.float64, torch.float32),
        pytest.param(16, torch.float32, torch.float16, marks=RunIf(min_gpus=1)),
        pytest.param("mixed", torch.float32, torch.float16, marks=RunIf(min_gpus=1)),
    ],
)
def test_lite_module_forward_conversion(precision, input_type, expected_type):
    lite = EmptyLite(precision=precision)

    def check_autocast(forward_input):
        assert precision not in (16, "mixed") or torch.is_autocast_enabled()
        return forward_input

    module = Mock(wraps=torch.nn.Linear(1, 1), side_effect=check_autocast)
    lite_module = _LiteModule(module, lite._accelerator)
    out = lite_module(torch.rand(1, dtype=input_type))
    assert module.call_args[0][0].dtype == expected_type
    assert out.dtype == expected_type
