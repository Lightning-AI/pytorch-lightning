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
import torch
from lightning.fabric.plugins.precision.double import DoublePrecision


def test_double_precision_forward_context():
    precision = DoublePrecision()
    assert torch.get_default_dtype() == torch.float32
    with precision.forward_context():
        assert torch.get_default_dtype() == torch.float64
    assert torch.get_default_dtype() == torch.float32


def test_convert_module():
    precision = DoublePrecision()
    module = torch.nn.Linear(2, 2)
    assert module.weight.dtype == module.bias.dtype == torch.float32
    module = precision.convert_module(module)
    assert module.weight.dtype == module.bias.dtype == torch.float64
