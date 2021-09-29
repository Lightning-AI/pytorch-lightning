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
from math import sqrt, isclose

import pytest
import torch
import torch.nn as nn

from pytorch_lightning.utilities import grad_norm


@pytest.mark.parametrize(
    "norm_type,expected",
    [
        # bug in for L0 "norm"? total norm should be 5
        (
            0,
            {"grad_0.0_norm_param0": 3, "grad_0.0_norm_param1": 2, "grad_0.0_norm_total": 2},
        ),
        (
            1,
            {"grad_1.0_norm_param0": 1 + 2 + 3, "grad_1.0_norm_param1": 4 + 5, "grad_1.0_norm_total": 15},
        ),
        (
            2,
            {
                "grad_2.0_norm_param0": pow(1 + 4 + 9, 0.5),
                "grad_2.0_norm_param1": pow(16 + 25, 0.5),
                "grad_2.0_norm_total": pow(1 + 4 + 9 + 16 + 25, 0.5),
            },
        ),
        (
            3.14,
            {
                "grad_3.14_norm_param0": pow(1 + 2 ** 3.14 + 3 ** 3.14, 1 / 3.14),
                "grad_3.14_norm_param1": pow(4 ** 3.14 + 5 ** 3.14, 1 / 3.14),
                "grad_3.14_norm_total": pow(1 + 2 ** 3.14 + 3 ** 3.14 + 4 ** 3.14 + 5 ** 3.14, 1 / 3.14),
            },
        ),
        (
            "inf",
            {
                "grad_inf_norm_param0": max(1, 2, 3),
                "grad_inf_norm_param1": max(4, 5),
                "grad_inf_norm_total": max(1, 2, 3, 4, 5),
            },
        ),
    ],
)
def test_grad_norm(norm_type, expected):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.param0 = nn.Parameter(torch.rand(3))
            self.param1 = nn.Parameter(torch.rand(2, 1))
            self.param0.grad = torch.tensor([-1.0, 2.0, -3.0])
            self.param1.grad = torch.tensor([[-4.0], [5.0]])

    model = Model()
    norms = grad_norm(model, norm_type)
    expected = {k: round(v, 4) for k, v in expected.items()}
    assert norms == expected
