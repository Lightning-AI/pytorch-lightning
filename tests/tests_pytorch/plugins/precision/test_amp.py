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
import threading
from unittest.mock import Mock, patch

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


@pytest.mark.parametrize(
    ("precision", "scaler", "should_error"),
    [
        ("16-mixed", Mock(), True),  # fp16 with scaler: fused optimizer + clip = error
        ("bf16-mixed", None, False),  # bf16 no scaler: fused optimizer + clip = ok
    ],
)
def test_optimizer_amp_scaling_support_in_step_method(precision, scaler, should_error):
    """Test that gradient clipping with fused optimizers is only blocked when a scaler is present.

    The `_step_supports_amp_scaling` flag indicates the optimizer handles unscaling internally (e.g., fused Adam).
    This is incompatible with gradient clipping only when using a GradScaler (16-mixed), since we can't unscale
    before clipping. With bf16-mixed there's no scaler, so gradient clipping works normally.

    """
    optimizer = Mock(_step_supports_amp_scaling=True)
    plugin = MixedPrecision(precision=precision, device="cuda:0", scaler=scaler)
    plugin.clip_grad_by_norm = Mock()

    if should_error:
        with pytest.raises(RuntimeError, match="The current optimizer.*does not allow for gradient clipping"):
            plugin.clip_gradients(optimizer, clip_val=1.0)
    else:
        plugin.clip_gradients(optimizer, clip_val=1.0, gradient_clip_algorithm=GradClipAlgorithmType.NORM)
        plugin.clip_grad_by_norm.assert_called_once()


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


@pytest.mark.parametrize("grad_mode_fn", [lambda: torch.no_grad(), lambda: torch.inference_mode()])
def test_amp_grad_mode_thread_isolation(grad_mode_fn):
    """Test that unrelated background threads entering grad-disabled contexts do not trigger cache clears."""
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")
    cleared_threads = []
    orig_clear = torch.clear_autocast_cache

    def spy_clear():
        cleared_threads.append(threading.get_ident())
        orig_clear()

    worker_started = threading.Event()
    worker_done = threading.Event()
    worker_error: list[Exception] = []
    worker_tid = None

    def worker():
        nonlocal worker_tid
        worker_tid = threading.get_ident()
        worker_started.set()
        try:
            with grad_mode_fn():
                _ = torch.ones(1)
        except Exception as e:
            worker_error.append(e)
        finally:
            worker_done.set()

    with patch("torch.clear_autocast_cache", side_effect=spy_clear), amp.forward_context():
        worker_thread = threading.Thread(target=worker, name="background-worker")
        worker_thread.start()
        assert worker_started.wait(timeout=5)
        assert worker_done.wait(timeout=5)
        worker_thread.join(timeout=5)
        assert not worker_thread.is_alive()
        assert not worker_error

        # Main thread in AMP does clear cache on exit of grad-disabling context
        main_cleared_before = len(cleared_threads)
        with grad_mode_fn():
            _ = torch.ones(1)
        assert len(cleared_threads) > main_cleared_before

    assert worker_tid is not None
    assert worker_tid not in cleared_threads


@pytest.mark.parametrize("ctx_factory", [lambda: torch.inference_mode(False), lambda: torch.inference_mode(mode=False)])
def test_amp_inference_mode_false_does_not_clear_cache(ctx_factory):
    """Test that inference_mode(mode=False) inside active AMP does not clear autocast cache."""
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")
    cleared_calls = 0
    orig_clear = torch.clear_autocast_cache

    def spy_clear():
        nonlocal cleared_calls
        cleared_calls += 1
        orig_clear()

    with patch("torch.clear_autocast_cache", side_effect=spy_clear), amp.forward_context():
        with ctx_factory():
            _ = torch.ones(1)
        assert cleared_calls == 0


def test_amp_forward_context_exception_cleanup():
    """Test that an exception inside forward_context properly restores patched functions and state."""
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")
    original_no_grad = torch.no_grad
    original_inference_mode = torch.inference_mode

    with pytest.raises(RuntimeError, match="controlled test failure"), amp.forward_context():
        assert torch.no_grad is not original_no_grad
        raise RuntimeError("controlled test failure")

    assert torch.no_grad is original_no_grad
    assert torch.inference_mode is original_inference_mode

    # Ensure subsequent no_grad calls in ordinary PyTorch work without issue
    with torch.no_grad():
        assert not torch.is_grad_enabled()


def test_amp_overlapping_forward_contexts_thread_safety():
    """Test that concurrent/overlapping AMP contexts across threads do not clobber each other's restoration."""
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")
    original_no_grad = torch.no_grad
    original_inference_mode = torch.inference_mode

    cleared_threads = []
    orig_clear = torch.clear_autocast_cache

    def spy_clear():
        cleared_threads.append(threading.get_ident())
        orig_clear()

    t1_entered = threading.Event()
    t2_entered = threading.Event()
    t1_can_exit = threading.Event()
    t2_can_exit = threading.Event()
    t1_exited = threading.Event()
    t2_cleared = threading.Event()
    errors: list[Exception] = []

    def run_t1():
        try:
            with amp.forward_context():
                t1_entered.set()
                assert t1_can_exit.wait(timeout=5)
        except Exception as e:
            errors.append(e)
        finally:
            t1_exited.set()

    def run_t2():
        try:
            assert t1_entered.wait(timeout=5)
            with amp.forward_context():
                t2_entered.set()
                assert t1_exited.wait(timeout=5)
                # T1 has exited, but T2 is still in forward_context, so no_grad should still clear cache
                before_len = len(cleared_threads)
                with torch.no_grad():
                    _ = torch.ones(1)
                assert len(cleared_threads) > before_len
                t2_cleared.set()
                assert t2_can_exit.wait(timeout=5)
        except Exception as e:
            errors.append(e)

    with patch("torch.clear_autocast_cache", side_effect=spy_clear):
        th1 = threading.Thread(target=run_t1)
        th2 = threading.Thread(target=run_t2)
        th1.start()
        th2.start()

        assert t2_entered.wait(timeout=5)
        t1_can_exit.set()
        assert t1_exited.wait(timeout=5)
        assert t2_cleared.wait(timeout=5)
        t2_can_exit.set()

        th1.join(timeout=5)
        th2.join(timeout=5)
        assert not th1.is_alive()
        assert not th2.is_alive()
        assert not errors

    assert torch.no_grad is original_no_grad
    assert torch.inference_mode is original_inference_mode


def test_amp_decorators_with_and_without_parentheses():
    """Test that @torch.no_grad and @torch.inference_mode work both with and without parentheses inside AMP."""
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")

    with amp.forward_context():

        @torch.no_grad()
        def fn_no_grad_parens():
            return torch.is_grad_enabled()

        @torch.no_grad
        def fn_no_grad_no_parens():
            return torch.is_grad_enabled()

        @torch.inference_mode()
        def fn_inf_parens():
            return torch.is_inference_mode_enabled()

        @torch.inference_mode
        def fn_inf_no_parens():
            return torch.is_inference_mode_enabled()

        assert fn_no_grad_parens() is False
        assert fn_no_grad_no_parens() is False
        assert fn_inf_parens() is True
        assert fn_inf_no_parens() is True


def test_amp_context_created_in_amp_exited_outside():
    """Test that a context manager created inside AMP does not clear cache if executed outside."""
    amp = MixedPrecision(precision="bf16-mixed", device="cpu")
    cleared = []
    orig_clear = torch.clear_autocast_cache

    def spy_clear():
        cleared.append(threading.get_ident())
        orig_clear()

    with patch("torch.clear_autocast_cache", side_effect=spy_clear):
        with amp.forward_context():
            ctx = torch.no_grad()

        # ctx exited outside forward_context()
        before = len(cleared)
        with ctx:
            _ = torch.ones(1)
        assert len(cleared) == before
