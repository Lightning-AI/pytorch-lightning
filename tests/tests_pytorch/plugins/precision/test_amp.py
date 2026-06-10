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
from torch import nn
from torch.optim import Optimizer

from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning.pytorch.utilities.imports import _TORCH_GREATER_EQUAL_2_11
from tests_pytorch.helpers.runif import RunIf


def test_clip_gradients():
    """Test that `.clip_gradients()` is a no-op when clipping is disabled."""
    optimizer = Mock(spec=Optimizer)
    precision = MixedPrecision(precision="16-mixed", device="cuda:0", scaler=Mock())
    precision.clip_grad_by_value = Mock()
    precision.clip_grad_by_norm = Mock()
    precision.clip_gradients(optimizer)
    precision.clip_grad_by_value.assert_not_called()
    precision.clip_grad_by_norm.assert_not_called()

    precision.clip_gradients(optimizer, clip_val=1.0, gradient_clip_algorithm=GradClipAlgorithmType.VALUE)
    precision.clip_grad_by_value.assert_called_once()
    precision.clip_grad_by_norm.assert_not_called()

    precision.clip_grad_by_value.reset_mock()
    precision.clip_grad_by_norm.reset_mock()

    precision.clip_gradients(optimizer, clip_val=1.0, gradient_clip_algorithm=GradClipAlgorithmType.NORM)
    precision.clip_grad_by_value.assert_not_called()
    precision.clip_grad_by_norm.assert_called_once()


def test_optimizer_amp_scaling_support_in_step_method():
    """Test that the plugin checks if the optimizer takes over unscaling in its step, making it incompatible with
    gradient clipping (example: fused Adam)."""

    optimizer = Mock(_step_supports_amp_scaling=True)
    precision = MixedPrecision(precision="16-mixed", device="cuda:0", scaler=Mock())

    with pytest.raises(RuntimeError, match="The current optimizer.*does not allow for gradient clipping"):
        precision.clip_gradients(optimizer, clip_val=1.0)


def test_amp_with_no_grad():
    """Test that asserts using `no_grad` context wrapper with a persistent AMP context wrapper does not break gradient
    tracking."""
    layer = nn.Linear(2, 1)
    x = torch.randn(1, 2)
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")

    with amp.forward_context():
        with torch.no_grad():
            _ = layer(x)

        loss = layer(x).mean()
        loss.backward()
        assert loss.grad_fn is not None


def test_amp_with_inference_mode():
    """Test that nested `inference_mode` also clears the autocast cache on exit."""
    layer = nn.Linear(2, 1)
    x = torch.randn(1, 2)
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")

    with amp.forward_context():
        with torch.inference_mode():
            _ = layer(x)

        loss = layer(x).mean()
        loss.backward()
        assert loss.grad_fn is not None


def test_amp_forward_context_restores_grad_mode_context_managers():
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")
    original_no_grad = torch.no_grad
    original_inference_mode = torch.inference_mode

    with amp.forward_context():
        assert torch.no_grad is not original_no_grad
        assert torch.inference_mode is not original_inference_mode

    assert torch.no_grad is original_no_grad
    assert torch.inference_mode is original_inference_mode


@pytest.mark.parametrize(("cache_enabled", "expect_grad"), [(True, False), (False, True)])
def test_torch_autocast_cache_behavior_with_no_grad(cache_enabled, expect_grad):
    """Document the underlying PyTorch autocast behavior that this plugin needs to handle."""
    # PyTorch 2.11 fixed the bug where the autocast cache retained the `no_grad` state correctly.
    # See: https://github.com/pytorch/pytorch/pull/165068
    if cache_enabled and _TORCH_GREATER_EQUAL_2_11:
        expect_grad = True
    layer = nn.Linear(2, 1)
    x = torch.randn(1, 2)

    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache_enabled):
        with torch.no_grad():
            _ = layer(x)

        loss = layer(x).mean()
        if expect_grad:
            loss.backward()
            assert loss.grad_fn is not None
        else:
            assert loss.grad_fn is None
            with pytest.raises(RuntimeError, match="does not require grad"):
                loss.backward()


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(("cache_enabled", "expect_grad"), [(True, False), (False, True)])
def test_torch_autocast_cache_behavior_with_no_grad_cuda(cache_enabled, expect_grad):
    """Document the same autocast cache behavior on CUDA, where the reported regression happens."""
    # PyTorch 2.11 fixed the bug where the autocast cache retained the `no_grad` state correctly.
    # See: https://github.com/pytorch/pytorch/pull/165068
    if cache_enabled and _TORCH_GREATER_EQUAL_2_11:
        expect_grad = True
    layer = nn.Linear(2, 1, device="cuda")
    x = torch.randn(1, 2, device="cuda")

    with torch.autocast("cuda", dtype=torch.float16, cache_enabled=cache_enabled):
        with torch.no_grad():
            _ = layer(x)

        loss = layer(x).mean()
        if expect_grad:
            loss.backward()
            assert loss.grad_fn is not None
        else:
            assert loss.grad_fn is None
            with pytest.raises(RuntimeError, match="does not require grad"):
                loss.backward()


@RunIf(min_cuda_gpus=1)
def test_amp_with_no_grad_cuda():
    """Test the Lightning workaround on the CUDA path used by the reported regression."""
    layer = nn.Linear(2, 1, device="cuda")
    x = torch.randn(1, 2, device="cuda")
    amp = MixedPrecision(precision="16-mixed", device="cuda")

    with amp.forward_context():
        with torch.no_grad():
            _ = layer(x)

        loss = layer(x).mean()
        loss.backward()
        assert loss.grad_fn is not None


def test_amp_autocast_context_manager_disables_cache():
    """Test that the public autocast context manager preserves the existing no-cache workaround."""
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")

    with amp.autocast_context_manager():
        assert not torch.is_autocast_cache_enabled()


def test_amp_forward_context_keeps_cache_enabled():
    """Test that Lightning's internal step context keeps the cached autocast path enabled."""
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")

    with amp.forward_context():
        assert torch.is_autocast_cache_enabled()
