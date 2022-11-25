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
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from tests_lite.helpers.runif import RunIf
from torch.optim import Adam

from lightning_lite.strategies import FSDPStrategy
from lightning_lite.strategies.fsdp import _FSDPBackwardSyncControl
from lightning_lite.utilities.imports import _TORCH_GREATER_EQUAL_1_12

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel, MixedPrecision


@mock.patch("lightning_lite.strategies.fsdp._TORCH_GREATER_EQUAL_1_12", False)
def test_fsdp_support(*_):
    with pytest.raises(NotImplementedError, match="`FSDPStrategy` is supported from PyTorch v1.12.0"):
        FSDPStrategy()


@RunIf(min_torch="1.12")
def test_fsdp_custom_mixed_precision(*_):
    """Test that passing a custom mixed precision config works."""
    config = MixedPrecision()
    strategy = FSDPStrategy(mixed_precision=config)
    assert strategy.mixed_precision_config == config


@RunIf(min_torch="1.12")
def test_fsdp_setup_optimizer_validation():
    """Test that `setup_optimizer()` validates the param groups and reference to FSDP parameters."""
    module = nn.Linear(2, 2)
    strategy = FSDPStrategy(parallel_devices=[torch.device("cpu")])

    bad_optimizer = Adam([{"params": [module.weight]}, {"params": [module.bias], "lr": 1e-3}])
    with pytest.raises(ValueError, match="does not support multiple param groups"):
        strategy.setup_optimizer(bad_optimizer)

    bad_optimizer = Adam(module.parameters())
    with pytest.raises(ValueError, match="The optimizer does not seem to reference any FSDP parameter"):
        strategy.setup_optimizer(bad_optimizer)


@RunIf(min_torch="1.12")
def test_fsdp_no_backward_sync():
    """Test that the backward sync control calls `.no_sync()`, and only on a module wrapped in
    FullyShardedDataParallel."""

    strategy = FSDPStrategy()
    assert isinstance(strategy._backward_sync_control, _FSDPBackwardSyncControl)

    with pytest.raises(
        TypeError, match="is only possible if the module passed to .* is wrapped in `FullyShardedDataParallel`"
    ):
        with strategy._backward_sync_control.no_backward_sync(Mock()):
            pass

    module = MagicMock(spec=FullyShardedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module):
        pass

    module.no_sync.assert_called_once()
