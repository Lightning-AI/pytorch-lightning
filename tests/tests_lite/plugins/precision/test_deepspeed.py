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
from unittest import mock
from unittest.mock import Mock

import pytest
from tests_lite.helpers.runif import RunIf

from lightning_lite.plugins.precision.deepspeed import DeepSpeedPrecision
from lightning_lite.utilities.types import Steppable


def test_invalid_precision_with_deepspeed_precision():
    with pytest.raises(ValueError, match="is not supported in DeepSpeed. `precision` must be one of"):
        DeepSpeedPrecision(precision=64, amp_type="native")


def test_deepspeed_precision_apex_not_installed(monkeypatch):
    import lightning_lite.plugins.precision.deepspeed as deepspeed

    monkeypatch.setattr(deepspeed, "_APEX_AVAILABLE", False)
    with pytest.raises(ImportError, match="You have asked for Apex AMP but `apex` is not installed."):
        DeepSpeedPrecision(precision=16, amp_type="apex")


@mock.patch("lightning_lite.plugins.precision.deepspeed._APEX_AVAILABLE", return_value=True)
def test_deepspeed_precision_apex_default_level(_):
    precision = DeepSpeedPrecision(precision=16, amp_type="apex")
    assert isinstance(precision, DeepSpeedPrecision)
    assert precision.amp_level == "O2"


def test_deepspeed_precision_backward():
    precision = DeepSpeedPrecision(precision=32, amp_type="native")
    tensor = Mock()
    model = Mock()
    precision.backward(tensor, model, "positional-arg", keyword="arg")
    model.backward.assert_called_once_with(tensor, "positional-arg", keyword="arg")


@RunIf(deepspeed=True)
def test_deepspeed_engine_is_steppable():
    """Test that the ``DeepSpeedEngine`` conforms to the Steppable API.

    If this fails, then optimization will be broken for DeepSpeed.
    """
    from deepspeed import DeepSpeedEngine

    engine = DeepSpeedEngine(Mock(), Mock())
    assert isinstance(engine, Steppable)


def test_deepspeed_precision_optimizer_step():
    precision = DeepSpeedPrecision(precision=32, amp_type="native")
    optimizer = model = Mock()
    precision.optimizer_step(optimizer, lr_kwargs=dict())
    model.step.assert_called_once_with(lr_kwargs=dict())
