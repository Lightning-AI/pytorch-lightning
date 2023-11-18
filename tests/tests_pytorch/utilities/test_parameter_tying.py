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
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities import find_shared_parameters, set_shared_parameters
from torch import nn


class ParameterSharingModule(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(32, 10, bias=False)
        self.layer_2 = nn.Linear(10, 32, bias=False)
        self.layer_3 = nn.Linear(32, 10, bias=False)
        self.layer_3.weight = self.layer_1.weight

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x


@pytest.mark.parametrize(
    ("model", "expected_shared_params"),
    [(BoringModel, []), (ParameterSharingModule, [["layer_1.weight", "layer_3.weight"]])],
)
def test_find_shared_parameters(model, expected_shared_params):
    assert expected_shared_params == find_shared_parameters(model())


def test_set_shared_parameters():
    model = ParameterSharingModule()
    set_shared_parameters(model, [["layer_1.weight", "layer_3.weight"]])

    assert torch.all(torch.eq(model.layer_1.weight, model.layer_3.weight))

    class SubModule(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return self.layer(x)

    class NestedModule(BoringModel):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(32, 10, bias=False)
            self.net_a = SubModule(self.layer)
            self.layer_2 = nn.Linear(10, 32, bias=False)
            self.net_b = SubModule(self.layer)

        def forward(self, x):
            x = self.net_a(x)
            x = self.layer_2(x)
            x = self.net_b(x)
            return x

    model = NestedModule()
    set_shared_parameters(model, [["layer.weight", "net_a.layer.weight", "net_b.layer.weight"]])

    assert torch.all(torch.eq(model.net_a.layer.weight, model.net_b.layer.weight))
