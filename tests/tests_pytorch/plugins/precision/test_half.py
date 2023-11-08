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

import pytest
import torch
from lightning.pytorch.plugins import HalfPrecision


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("bf16-true", torch.bfloat16),
        ("16-true", torch.half),
    ],
)
def test_selected_dtype(precision, expected_dtype):
    plugin = HalfPrecision(precision=precision)
    assert plugin.precision == precision
    assert plugin._desired_input_dtype == expected_dtype


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("bf16-true", torch.bfloat16),
        ("16-true", torch.half),
    ],
)
def test_module_init_context(precision, expected_dtype):
    plugin = HalfPrecision(precision=precision)
    with plugin.module_init_context():
        model = torch.nn.Linear(2, 2)
        assert torch.get_default_dtype() == expected_dtype
    assert model.weight.dtype == expected_dtype


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("bf16-true", torch.bfloat16),
        ("16-true", torch.half),
    ],
)
def test_forward_context(precision, expected_dtype):
    precision = HalfPrecision(precision=precision)
    assert torch.get_default_dtype() == torch.float32
    with precision.forward_context():
        assert torch.get_default_dtype() == expected_dtype
    assert torch.get_default_dtype() == torch.float32


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("bf16-true", torch.bfloat16),
        ("16-true", torch.half),
    ],
)
def test_convert_module(precision, expected_dtype):
    precision = HalfPrecision(precision=precision)
    module = torch.nn.Linear(2, 2)
    assert module.weight.dtype == module.bias.dtype == torch.float32
    module = precision.convert_module(module)
    assert module.weight.dtype == module.bias.dtype == expected_dtype
