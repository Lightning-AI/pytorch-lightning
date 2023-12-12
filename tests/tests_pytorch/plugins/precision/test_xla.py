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
import re
from unittest import mock
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch.plugins import XLAPrecision

from tests_pytorch.helpers.runif import RunIf


@RunIf(tpu=True)
@mock.patch.dict(os.environ, {}, clear=True)
def test_optimizer_step_calls_mark_step():
    plugin = XLAPrecision(precision="32-true")
    optimizer = Mock()
    with mock.patch("torch_xla.core.xla_model") as xm_mock:
        plugin.optimizer_step(optimizer=optimizer, model=Mock(), closure=Mock())
    optimizer.step.assert_called_once()
    xm_mock.mark_step.assert_called_once()


@mock.patch.dict(os.environ, {}, clear=True)
def test_precision_input_validation(xla_available):
    XLAPrecision(precision="32-true")
    XLAPrecision(precision="16-true")
    XLAPrecision(precision="bf16-true")

    with pytest.raises(ValueError, match=re.escape("`precision='16')` is not supported in XLA")):
        XLAPrecision("16")
    with pytest.raises(ValueError, match=re.escape("`precision='16-mixed')` is not supported in XLA")):
        XLAPrecision("16-mixed")
    with pytest.raises(ValueError, match=re.escape("`precision='bf16-mixed')` is not supported in XLA")):
        XLAPrecision("bf16-mixed")
    with pytest.raises(ValueError, match=re.escape("`precision='64-true')` is not supported in XLA")):
        XLAPrecision("64-true")


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("bf16-true", torch.bfloat16),
        ("16-true", torch.half),
    ],
)
@mock.patch.dict(os.environ, {}, clear=True)
def test_selected_dtype(precision, expected_dtype, xla_available):
    plugin = XLAPrecision(precision=precision)
    assert plugin.precision == precision
    assert plugin._desired_dtype == expected_dtype


def test_teardown(xla_available):
    plugin = XLAPrecision(precision="16-true")
    assert os.environ["XLA_USE_F16"] == "1"
    plugin.teardown()
    assert "XLA_USE_B16" not in os.environ

    plugin = XLAPrecision(precision="bf16-true")
    assert os.environ["XLA_USE_BF16"] == "1"
    plugin.teardown()
    assert "XLA_USE_BF16" not in os.environ
