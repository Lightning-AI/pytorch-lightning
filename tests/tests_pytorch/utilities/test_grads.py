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
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from lightning.pytorch.utilities import grad_norm


@pytest.mark.parametrize(
    ("norm_type", "expected"),
    [
        (
            1,
            {"grad_1.0_norm/param0": 1 + 2 + 3, "grad_1.0_norm/param1": 4 + 5, "grad_1.0_norm_total": 15.0},
        ),
        (
            2,
            {
                "grad_2.0_norm/param0": pow(1 + 4 + 9, 0.5),
                "grad_2.0_norm/param1": pow(16 + 25, 0.5),
                "grad_2.0_norm_total": pow(1 + 4 + 9 + 16 + 25, 0.5),
            },
        ),
        (
            3.14,
            {
                "grad_3.14_norm/param0": pow(1 + 2**3.14 + 3**3.14, 1 / 3.14),
                "grad_3.14_norm/param1": pow(4**3.14 + 5**3.14, 1 / 3.14),
                "grad_3.14_norm_total": pow(1 + 2**3.14 + 3**3.14 + 4**3.14 + 5**3.14, 1 / 3.14),
            },
        ),
        (
            "inf",
            {
                "grad_inf_norm/param0": max(1, 2, 3),
                "grad_inf_norm/param1": max(4, 5),
                "grad_inf_norm_total": max(1, 2, 3, 4, 5),
            },
        ),
    ],
)
def test_grad_norm(norm_type, expected):
    """Test utility function for computing the p-norm of individual parameter groups and norm in total."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.param0 = nn.Parameter(torch.rand(3))
            self.param1 = nn.Parameter(torch.rand(2, 1))
            self.param0.grad = torch.tensor([-1.0, 2.0, -3.0])
            self.param1.grad = torch.tensor([[-4.0], [5.0]])
            # param without grad should not contribute to norm
            self.param2 = nn.Parameter(torch.rand(1))

    model = Model()
    norms = grad_norm(model, norm_type)

    assert norms.keys() == expected.keys()
    for k in norms:
        assert norms[k] == pytest.approx(expected[k])


@pytest.mark.parametrize("norm_type", [-1, 0])
def test_grad_norm_invalid_norm_type(norm_type):
    with pytest.raises(ValueError, match="`norm_type` must be a positive number or 'inf'"):
        grad_norm(Mock(), norm_type)


def test_grad_norm_with_double_dtype():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            dtype = torch.double
            self.param = nn.Parameter(torch.tensor(1.0, dtype=dtype))
            # grad norm of this would become infinite
            self.param.grad = torch.tensor(1e23, dtype=dtype)

    model = Model()
    norms = grad_norm(model, 2)
    assert all(torch.isfinite(torch.tensor(v)) for v in norms.values()), norms
