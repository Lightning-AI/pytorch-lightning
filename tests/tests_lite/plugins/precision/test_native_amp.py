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
import torch

from lightning_lite.plugins.precision.native_amp import NativeMixedPrecisionPlugin


def test_native_amp_precision_default_scaler():
    precision_plugin = NativeMixedPrecisionPlugin(precision=16, device=Mock())
    assert isinstance(precision_plugin.scaler, torch.cuda.amp.GradScaler)


@mock.patch("lightning_lite.plugins.precision.native_amp._TORCH_GREATER_EQUAL_1_10", True)
def test_native_amp_precision_scaler_with_bf16():
    with pytest.raises(ValueError, match="`precision='bf16'` does not use a scaler"):
        NativeMixedPrecisionPlugin(precision="bf16", device=Mock(), scaler=Mock())

    precision_plugin = NativeMixedPrecisionPlugin(precision="bf16", device=Mock())
    assert precision_plugin.scaler is None


@mock.patch("lightning_lite.plugins.precision.native_amp._TORCH_GREATER_EQUAL_1_10", False)
def test_native_amp_precision_bf16_min_torch():
    with pytest.raises(ImportError, match="you must install torch greater or equal to 1.10"):
        NativeMixedPrecisionPlugin(precision="bf16", device=Mock())


def test_native_amp_precision_forward_context():
    precision_plugin = NativeMixedPrecisionPlugin(precision="mixed", device="cuda")
    assert torch.get_default_dtype() == torch.float32
    with precision_plugin.forward_context():
        assert torch.get_autocast_gpu_dtype() == torch.float16


def test_native_amp_precision_backward():
    precision_plugin = NativeMixedPrecisionPlugin(precision="mixed", device="cuda")
    precision_plugin.scaler = Mock()
    precision_plugin.scaler.scale = Mock(side_effect=(lambda x: x))
    tensor = Mock()
    model = Mock()
    precision_plugin.backward(tensor, model, "positional-arg", keyword="arg")
    precision_plugin.scaler.scale.assert_called_once_with(tensor)
    tensor.backward.assert_called_once_with("positional-arg", keyword="arg")


def test_native_amp_precision_optimizer_step_with_scaler():
    precision_plugin = NativeMixedPrecisionPlugin(precision="mixed", device="cuda")
    precision_plugin.scaler = Mock()
    optimizer = Mock()
    model = Mock()

    precision_plugin.optimizer_step(optimizer, "positional-arg", model=model, keyword="arg")
    precision_plugin.scaler.unscale_.assert_called_once_with(optimizer)
    precision_plugin.scaler.step.assert_called_once_with(optimizer, "positional-arg", keyword="arg")
    precision_plugin.scaler.update.assert_called_once()


def test_native_amp_precision_optimizer_step_without_scaler():
    precision_plugin = NativeMixedPrecisionPlugin(precision="bf16", device="cuda")
    assert precision_plugin.scaler is None
    optimizer = Mock()
    model = Mock()

    precision_plugin.optimizer_step(optimizer, "positional-arg", model=model, keyword="arg")
    optimizer.step.assert_called_once_with("positional-arg", keyword="arg")
