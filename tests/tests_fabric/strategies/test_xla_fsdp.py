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
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from lightning.fabric import Fabric
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.strategies import XLAFSDPStrategy
from lightning.fabric.strategies.xla_fsdp import _XLAFSDPBackwardSyncControl
from tests_fabric.helpers.runif import RunIf


@RunIf(min_torch="2.0", tpu=True)
@pytest.mark.parametrize("torch_ge_2_0", [False, True])
def test_xla_fsdp_setup_optimizer_validation(torch_ge_2_0):
    """Test that `setup_optimizer()` validates the param groups and reference to FSDP parameters."""
    module = nn.Linear(2, 2)
    strategy = XLAFSDPStrategy(
        parallel_devices=XLAAccelerator.get_parallel_devices(XLAAccelerator.auto_device_count()),
    )

    with mock.patch("lightning.fabric.strategies.xla_fsdp._TORCH_GREATER_EQUAL_2_0", torch_ge_2_0):
        bad_optimizer_1 = Adam([{"params": [module.weight]}, {"params": [module.bias], "lr": 1e-3}])
        bad_optimizer_2 = Adam(module.parameters())

        if torch_ge_2_0:
            strategy.setup_optimizer(bad_optimizer_1)
            strategy.setup_optimizer(bad_optimizer_2)
        else:
            with pytest.raises(ValueError, match="does not support multiple param groups"):
                strategy.setup_optimizer(bad_optimizer_1)
            with pytest.raises(ValueError, match="The optimizer does not seem to reference any XLAFSDP parameter"):
                strategy.setup_optimizer(bad_optimizer_2)


@RunIf(min_torch="2.0", tpu=True)
def test_xla_fsdp_no_backward_sync():
    """Test that the backward sync control calls `.no_sync()`, and only on a module wrapped in
    XlaFullyShardedDataParallel."""
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

    strategy = XLAFSDPStrategy()
    assert isinstance(strategy._backward_sync_control, _XLAFSDPBackwardSyncControl)

    with pytest.raises(
        TypeError, match="is only possible if the module passed to .* is wrapped in `XlaFullyShardedDataParallel`"
    ), strategy._backward_sync_control.no_backward_sync(object()):
        pass

    module = MagicMock(spec=XlaFullyShardedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module):
        pass

    module.no_sync.assert_called_once()


@RunIf(min_torch="2.0", tpu=True)
def test_xla_fsdp_grad_clipping_value_error():
    strategy = XLAFSDPStrategy()
    with pytest.raises(NotImplementedError, match="does not support to clip gradients by value"):
        strategy.clip_gradients_value(Mock(), Mock(), Mock())


@RunIf(min_torch="2.0", tpu=True)
def test_xla_fsdp_activation_checkpointing_setup():
    """Test XLAFSDP activation checkpointing setup."""
    from torch_xla.distributed.fsdp import checkpoint_module
    from torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel

    auto_wrapper_callable = lambda m, *args, **kwargs: XlaFullyShardedDataParallel(
        checkpoint_module(m), *args, **kwargs
    )
    strategy = XLAFSDPStrategy(auto_wrapper_callable=auto_wrapper_callable)

    assert auto_wrapper_callable in strategy._fsdp_kwargs.values()


def xla_fsdp_rewrap_warning(fabric: Fabric):
    """Fabric launch function for test_xla_fsdp_rewrap_warning."""
    from torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel

    with fabric.init_module():
        model = torch.nn.Sequential(
            torch.nn.Linear(1, 1), torch.nn.ReLU(), XlaFullyShardedDataParallel(torch.nn.Linear(1, 1))
        )
    if fabric.node_rank:
        with pytest.warns(match="submodule is already wrapped"):
            model = fabric.setup_module(model)
    else:
        model = fabric.setup_module(model)
    fabric.barrier("warning_check")
    assert not isinstance(model._forward_module[0], XlaFullyShardedDataParallel)
    assert not isinstance(model._forward_module[1], XlaFullyShardedDataParallel)
    assert isinstance(model._forward_module[2], XlaFullyShardedDataParallel)


@RunIf(min_torch="2.0", tpu=True, standalone=True)
def test_xla_fsdp_rewrap_warning():
    """Test XLAFSDP rewrap warning."""
    from torch_xla.distributed.fsdp.wrap import always_wrap_policy

    strategy = XLAFSDPStrategy(auto_wrap_policy=always_wrap_policy)
    fabric = Fabric(accelerator="tpu", strategy=strategy)
    fabric.launch(xla_fsdp_rewrap_warning)
