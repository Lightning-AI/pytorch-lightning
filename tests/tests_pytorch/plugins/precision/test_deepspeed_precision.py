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
from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecision


def test_invalid_precision_with_deepspeed_precision():
    with pytest.raises(ValueError, match="is not supported. `precision` must be one of"):
        DeepSpeedPrecision(precision="64-true")


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
        ("bf16-mixed", torch.bfloat16),
        ("16-mixed", torch.float16),
        ("bf16-true", torch.bfloat16),
        ("16-true", torch.float16),
    ],
)
def test_selected_dtype(precision, expected_dtype):
    plugin = DeepSpeedPrecision(precision=precision)
    assert plugin.precision == precision
    assert plugin._desired_dtype == expected_dtype


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
        ("bf16-mixed", torch.float32),
        ("16-mixed", torch.float32),
        ("bf16-true", torch.bfloat16),
        ("16-true", torch.float16),
    ],
)
def test_module_init_context(precision, expected_dtype):
    plugin = DeepSpeedPrecision(precision=precision)
    with plugin.module_init_context():
        model = torch.nn.Linear(2, 2)
        assert torch.get_default_dtype() == expected_dtype
    assert model.weight.dtype == expected_dtype


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("32-true", torch.float32),
        ("bf16-mixed", torch.float32),
        ("16-mixed", torch.float32),
        ("bf16-true", torch.bfloat16),
        ("16-true", torch.float16),
    ],
)
def test_convert_module(precision, expected_dtype):
    precision = DeepSpeedPrecision(precision=precision)
    module = torch.nn.Linear(2, 2)
    assert module.weight.dtype == module.bias.dtype == torch.float32
    module = precision.convert_module(module)
    assert module.weight.dtype == module.bias.dtype == expected_dtype
