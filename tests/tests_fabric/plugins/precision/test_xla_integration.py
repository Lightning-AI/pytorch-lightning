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
import os
from unittest import mock

import pytest
import torch
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.plugins import XLAPrecision

from tests_fabric.helpers.runif import RunIf


class BoringPrecisionModule(nn.Module):
    def __init__(self, expected_dtype):
        super().__init__()
        self.expected_dtype = expected_dtype
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        # TODO: These should be float16/bfloat16
        assert x.dtype == torch.float32
        assert torch.tensor([0.0]).dtype == torch.float32
        return self.layer(x)


def _run_xla_precision(fabric, expected_dtype):
    with fabric.init_module():
        model = BoringPrecisionModule(expected_dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model, optimizer = fabric.setup(model, optimizer)

    batch = torch.rand(2, 32, device=fabric.device)

    # TODO: This should be float16/bfloat16
    assert model.layer.weight.dtype == model.layer.bias.dtype == torch.float32

    assert batch.dtype == torch.float32
    output = model(batch)
    assert output.dtype == torch.float32
    loss = torch.nn.functional.mse_loss(output, torch.ones_like(output))
    fabric.backward(loss)
    assert model.layer.weight.grad.dtype == torch.float32
    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.parametrize(("precision", "expected_dtype"), [("16-true", torch.float16), ("bf16-true", torch.bfloat16)])
@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_xla_precision(precision, expected_dtype):
    fabric = Fabric(devices=1, precision=precision)
    assert isinstance(fabric._precision, XLAPrecision)
    fabric.launch(_run_xla_precision, expected_dtype=expected_dtype)
