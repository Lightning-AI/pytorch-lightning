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
from tests_lite.helpers.runif import RunIf

from lightning_lite.plugins.precision.native_amp import NativeMixedPrecision


def test_native_amp_precision_default_scaler():
    precision = NativeMixedPrecision(precision=16, device=Mock())
    assert isinstance(precision.scaler, torch.cuda.amp.GradScaler)


@mock.patch("lightning_lite.plugins.precision.native_amp._TORCH_GREATER_EQUAL_1_10", True)
def test_native_amp_precision_scaler_with_bf16():
    with pytest.raises(ValueError, match="`precision='bf16'` does not use a scaler"):
        NativeMixedPrecision(precision="bf16", device=Mock(), scaler=Mock())

    precision = NativeMixedPrecision(precision="bf16", device=Mock())
    assert precision.scaler is None


@mock.patch("lightning_lite.plugins.precision.native_amp._TORCH_GREATER_EQUAL_1_10", False)
def test_native_amp_precision_bf16_min_torch():
    with pytest.raises(ImportError, match="you must install torch greater or equal to 1.10"):
        NativeMixedPrecision(precision="bf16", device=Mock())


@RunIf(min_torch="1.10")
def test_native_amp_precision_forward_context():
    """Test to ensure that the context manager correctly is set to CPU + bfloat16."""
    precision = NativeMixedPrecision(precision=16, device="cuda")
    assert precision.device == "cuda"
    assert isinstance(precision.scaler, torch.cuda.amp.GradScaler)
    assert torch.get_default_dtype() == torch.float32
    with precision.forward_context():
        assert torch.get_autocast_gpu_dtype() == torch.float16

    precision = NativeMixedPrecision(precision="bf16", device="cpu")
    assert precision.device == "cpu"
    assert precision.scaler is None
    with precision.forward_context():
        assert torch.get_autocast_cpu_dtype() == torch.bfloat16

    context_manager = precision._autocast_context_manager()
    assert isinstance(context_manager, torch.autocast)
    # check with str due to a bug upstream: https://github.com/pytorch/pytorch/issues/65786
    assert str(context_manager.fast_dtype) == str(torch.bfloat16)


def test_native_amp_precision_backward():
    precision = NativeMixedPrecision(precision="mixed", device="cuda")
    precision.scaler = Mock()
    precision.scaler.scale = Mock(side_effect=(lambda x: x))
    tensor = Mock()
    model = Mock()
    precision.backward(tensor, model, "positional-arg", keyword="arg")
    precision.scaler.scale.assert_called_once_with(tensor)
    tensor.backward.assert_called_once_with("positional-arg", keyword="arg")


def test_native_amp_precision_optimizer_step_with_scaler():
    precision = NativeMixedPrecision(precision="mixed", device="cuda")
    precision.scaler = Mock()
    optimizer = Mock()

    precision.optimizer_step(optimizer, keyword="arg")
    precision.scaler.step.assert_called_once_with(optimizer, keyword="arg")
    precision.scaler.update.assert_called_once()


@RunIf(min_torch="1.10")
def test_native_amp_precision_optimizer_step_without_scaler():
    precision = NativeMixedPrecision(precision="bf16", device="cuda")
    assert precision.scaler is None
    optimizer = Mock()

    precision.optimizer_step(optimizer, keyword="arg")
    optimizer.step.assert_called_once_with(keyword="arg")
