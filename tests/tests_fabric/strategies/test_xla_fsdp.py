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
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
import torch.nn
import torch.nn as nn
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.plugins import XLAPrecision
from lightning.fabric.strategies import XLAFSDPStrategy
from lightning.fabric.strategies.xla_fsdp import _activation_checkpointing_auto_wrapper, _XLAFSDPBackwardSyncControl
from torch.optim import Adam

from tests_fabric.helpers.runif import RunIf


@RunIf(tpu=True)
def test_xla_fsdp_setup_optimizer_validation():
    """Test that `setup_optimizer()` validates the param groups and reference to FSDP parameters."""
    module = nn.Linear(2, 2)
    strategy = XLAFSDPStrategy(
        parallel_devices=XLAAccelerator.get_parallel_devices(XLAAccelerator.auto_device_count()),
    )
    bad_optimizer = Adam(module.parameters())
    with pytest.raises(ValueError, match="The optimizer does not seem to reference any XLAFSDP parameter"):
        strategy.setup_optimizer(bad_optimizer)


@RunIf(tpu=True)
def test_xla_fsdp_no_backward_sync():
    """Test that the backward sync control calls `.no_sync()`, and only on a module wrapped in
    XlaFullyShardedDataParallel."""
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

    strategy = XLAFSDPStrategy()
    assert isinstance(strategy._backward_sync_control, _XLAFSDPBackwardSyncControl)

    with pytest.raises(
        TypeError, match="is only possible if the module passed to .* is wrapped in `XlaFullyShardedDataParallel`"
    ), strategy._backward_sync_control.no_backward_sync(object(), True):
        pass

    module = MagicMock(spec=XlaFullyShardedDataParallel)

    with strategy._backward_sync_control.no_backward_sync(module, False):
        pass
    module.no_sync.assert_not_called()

    with strategy._backward_sync_control.no_backward_sync(module, True):
        pass
    module.no_sync.assert_called_once()


@RunIf(tpu=True)
def test_xla_fsdp_grad_clipping_value_error():
    strategy = XLAFSDPStrategy()
    with pytest.raises(NotImplementedError, match="does not support to clip gradients by value"):
        strategy.clip_gradients_value(Mock(), Mock(), Mock())


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_rank_properties_access(xla_available):
    """Test that the strategy returns the expected values depending on whether we're in the main process or not."""
    strategy = XLAFSDPStrategy()
    strategy.cluster_environment = Mock()

    # we're in the main process, no processes have been launched yet
    assert not strategy._launched
    assert strategy.global_rank == 0
    assert strategy.local_rank == 0
    assert strategy.node_rank == 0
    assert strategy.world_size == 1

    # simulate we're in a worker process
    strategy._launched = True
    assert strategy.global_rank == strategy.cluster_environment.global_rank()
    assert strategy.local_rank == strategy.cluster_environment.local_rank()
    assert strategy.node_rank == strategy.cluster_environment.node_rank()
    assert strategy.world_size == strategy.cluster_environment.world_size()


def test_xla_fsdp_policy(xla_available):
    strategy = XLAFSDPStrategy(foo=1)
    assert strategy._fsdp_kwargs == {"foo": 1}

    strategy = XLAFSDPStrategy(auto_wrap_policy={torch.nn.Linear})
    kwargs = strategy._parse_fsdp_kwargs()
    assert set(kwargs) == {"auto_wrap_policy", "compute_dtype"}
    assert kwargs["auto_wrap_policy"].func._mock_name == "transformer_auto_wrap_policy"
    assert kwargs["compute_dtype"] is torch.float32

    strategy = XLAFSDPStrategy(activation_checkpointing_policy={torch.nn.Linear})
    _ = strategy._parse_fsdp_kwargs()
    kwargs = strategy._parse_fsdp_kwargs()  # ensure it's idempotent
    assert set(kwargs) == {"auto_wrapper_callable", "compute_dtype"}
    assert kwargs["auto_wrapper_callable"].func is _activation_checkpointing_auto_wrapper
    assert kwargs["compute_dtype"] is torch.float32

    strategy = XLAFSDPStrategy(
        accelerator=Mock(),
        auto_wrap_policy={torch.nn.Linear},
        activation_checkpointing_policy={torch.nn.Linear},
        precision=XLAPrecision("bf16-true"),
    )
    kwargs = strategy._parse_fsdp_kwargs()
    assert set(kwargs) == {"auto_wrap_policy", "auto_wrapper_callable", "compute_dtype"}
    assert kwargs["auto_wrap_policy"].func._mock_name == "transformer_auto_wrap_policy"
    assert kwargs["auto_wrapper_callable"].func is _activation_checkpointing_auto_wrapper
    assert kwargs["compute_dtype"] is torch.bfloat16
    strategy.teardown()

    strategy = XLAFSDPStrategy(activation_checkpointing_policy={torch.nn.Linear}, auto_wrapper_callable="foo")
    with pytest.raises(ValueError, match="cannot set both"):
        strategy._parse_fsdp_kwargs()

    strategy = XLAFSDPStrategy(activation_checkpointing_policy="foo")
    with pytest.raises(TypeError, match="must be a set"):
        strategy._parse_fsdp_kwargs()
