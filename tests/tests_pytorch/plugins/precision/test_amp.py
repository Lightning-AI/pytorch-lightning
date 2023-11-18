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
from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.utilities import GradClipAlgorithmType
from torch.optim import Optimizer


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
