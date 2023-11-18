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
"""Integration tests for double-precision training."""

import torch
import torch.nn as nn
from lightning.fabric import Fabric

from tests_fabric.helpers.runif import RunIf


class BoringDoubleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)
        self.register_buffer("complex_buffer", torch.complex(torch.rand(10), torch.rand(10)), False)

    def forward(self, x):
        assert x.dtype == torch.float64
        # the default dtype for new tensors is now float64
        assert torch.tensor([0.0]).dtype == torch.float64
        return self.layer(x)


@RunIf(mps=False)  # MPS doesn't support float64
def test_double_precision():
    fabric = Fabric(devices=1, precision="64-true")

    with fabric.init_module():
        model = BoringDoubleModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    batch = torch.rand(2, 32, device=fabric.device)
    assert model.layer.weight.dtype == model.layer.bias.dtype == torch.float64
    assert model.complex_buffer.dtype == torch.complex128

    assert batch.dtype == torch.float32
    output = model(batch)
    assert output.dtype == torch.float32
    loss = torch.nn.functional.mse_loss(output, torch.ones_like(output))
    fabric.backward(loss)
    assert model.layer.weight.grad.dtype == torch.float64
    optimizer.step()
    optimizer.zero_grad()
