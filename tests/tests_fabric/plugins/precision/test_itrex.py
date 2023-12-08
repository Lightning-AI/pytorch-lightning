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
# limitations under the License
import pytest
import torch
import torch.distributed
from lightning.fabric import Fabric
from lightning.fabric.plugins.precision.itrex import _ITREX_AVAILABLE, ITREXPrecision


@pytest.mark.skipif(not _ITREX_AVAILABLE, reason="intel-extension-for-transformers unavailable")
@pytest.mark.parametrize(
    ("mode", "expected_device", "expected_dtype"),
    [
        ("int8", "cpu", torch.int8),
        ("int4_fullrange", "cpu", torch.int8),
        ("nf4", "cpu", torch.int8),
        ("fp4_e2m1_bnb", "cpu", torch.int8),
    ],
)
def test_itrex_layers(mode, expected_device, expected_dtype):
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(2, 2)
            self.ln = torch.nn.LayerNorm(2)

    fabric = Fabric(devices=1, plugins=ITREXPrecision(mode=mode))
    with fabric.init_module():
        model = MyModel()
    assert model.l.weight.device.type == "cpu"
    assert model.l.weight.dtype == torch.float32
    # quantize
    model = fabric.setup(model)
    assert model.l.weight.device.type == expected_device
    assert model.l.weight.dtype == expected_dtype
