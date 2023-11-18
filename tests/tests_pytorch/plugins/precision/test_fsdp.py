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
from unittest.mock import ANY, MagicMock, Mock

import pytest
import torch
from lightning.fabric.plugins.precision.utils import _DtypeContextManager
from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision

from tests_pytorch.helpers.runif import RunIf


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        ("16-true", (torch.float16, torch.float16, torch.float16)),
        ("bf16-true", (torch.bfloat16, torch.bfloat16, torch.bfloat16)),
        pytest.param(
            "16-mixed", (torch.float32, torch.float16, torch.float16), marks=RunIf(min_torch="2.0"), id="16-mixed-ge2_0"
        ),
        pytest.param(
            "16-mixed", (None, torch.float16, torch.float16), marks=RunIf(max_torch="2.0"), id="16-mixed-lt2_0"
        ),
        pytest.param(
            "bf16-mixed",
            (torch.float32, torch.bfloat16, torch.bfloat16),
            marks=RunIf(min_torch="2.0"),
            id="bf16-mixed-ge2_0",
        ),
        pytest.param(
            "bf16-mixed", (None, torch.bfloat16, torch.bfloat16), marks=RunIf(max_torch="2.0"), id="bf16-mixed-lt2_0"
        ),
        pytest.param(
            "32-true", (torch.float32, torch.float32, torch.float32), marks=RunIf(min_torch="2.0"), id="32-true-ge2_0"
        ),
        pytest.param("32-true", (None, torch.float32, torch.float32), marks=RunIf(max_torch="2.0"), id="32-true-lt2_0"),
    ],
)
def test_fsdp_precision_config(precision, expected):
    plugin = FSDPPrecision(precision=precision)
    config = plugin.mixed_precision_config

    assert config.param_dtype == expected[0]
    assert config.buffer_dtype == expected[1]
    assert config.reduce_dtype == expected[2]


def test_fsdp_precision_default_scaler():
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

    precision = FSDPPrecision(precision="16-mixed")
    assert isinstance(precision.scaler, ShardedGradScaler)


def test_fsdp_precision_scaler_with_bf16():
    with pytest.raises(ValueError, match="`precision='bf16-mixed'` does not use a scaler"):
        FSDPPrecision(precision="bf16-mixed", scaler=Mock())

    precision = FSDPPrecision(precision="bf16-mixed")
    assert precision.scaler is None


@RunIf(min_cuda_gpus=1)
def test_fsdp_precision_forward_context():
    """Test to ensure that the context manager correctly is set to bfloat16."""
    precision = FSDPPrecision(precision="16-mixed")
    assert isinstance(precision.scaler, torch.cuda.amp.GradScaler)
    assert torch.get_default_dtype() == torch.float32
    with precision.forward_context():
        assert torch.get_autocast_gpu_dtype() == torch.float16
    assert isinstance(precision.forward_context(), torch.autocast)
    assert precision.forward_context().fast_dtype == torch.float16

    precision = FSDPPrecision(precision="16-true")
    assert precision.scaler is None
    assert torch.get_default_dtype() == torch.float32
    with precision.forward_context():
        assert torch.get_default_dtype() == torch.float16
    assert isinstance(precision.forward_context(), _DtypeContextManager)
    assert precision.forward_context()._new_dtype == torch.float16

    precision = FSDPPrecision(precision="bf16-mixed")
    assert precision.scaler is None
    with precision.forward_context():
        assert torch.get_autocast_gpu_dtype() == torch.bfloat16
    assert isinstance(precision.forward_context(), torch.autocast)
    assert precision.forward_context().fast_dtype == torch.bfloat16

    precision = FSDPPrecision(precision="bf16-true")
    assert precision.scaler is None
    with precision.forward_context():  # forward context is not using autocast ctx manager
        assert torch.get_default_dtype() == torch.bfloat16
    assert isinstance(precision.forward_context(), _DtypeContextManager)
    assert precision.forward_context()._new_dtype == torch.bfloat16


def test_fsdp_precision_backward():
    precision = FSDPPrecision(precision="16-mixed")
    precision.scaler = Mock()
    precision.scaler.scale = Mock(side_effect=(lambda x: x))
    tensor = Mock()
    model = Mock(trainer=Mock(callbacks=[], profiler=MagicMock()))
    precision.pre_backward(tensor, model)
    precision.backward(tensor, model, None, "positional-arg", keyword="arg")
    precision.scaler.scale.assert_called_once_with(tensor)
    model.backward.assert_called_once_with(tensor, "positional-arg", keyword="arg")


def test_fsdp_precision_optimizer_step_with_scaler():
    precision = FSDPPrecision(precision="16-mixed")
    precision.scaler = Mock()
    model = Mock(trainer=Mock(callbacks=[], profiler=MagicMock()))
    optimizer = Mock()
    closure = Mock()

    precision.optimizer_step(optimizer, model, closure, keyword="arg")
    precision.scaler.step.assert_called_once_with(optimizer, keyword="arg")
    precision.scaler.update.assert_called_once()


def test_fsdp_precision_optimizer_step_without_scaler():
    precision = FSDPPrecision(precision="bf16-mixed")
    assert precision.scaler is None
    model = Mock(trainer=Mock(callbacks=[], profiler=MagicMock()))
    optimizer = Mock()
    closure = Mock()

    precision.optimizer_step(optimizer, model, closure, keyword="arg")
    optimizer.step.assert_called_once_with(closure=ANY, keyword="arg")


def test_invalid_precision_with_fsdp_precision():
    FSDPPrecision("16-mixed")
    FSDPPrecision("bf16-mixed")

    with pytest.raises(ValueError, match="is not supported in FSDP. `precision` must be one of"):
        FSDPPrecision(precision="64-true")
