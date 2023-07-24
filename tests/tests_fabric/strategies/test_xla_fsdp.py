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
from copy import deepcopy
from unittest import mock
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from lightning.fabric import Fabric
from lightning.fabric.accelerators import XLAAccelerator
from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.strategies import XLAFSDPStrategy
from lightning.fabric.strategies.xla_fsdp import _XLAFSDPBackwardSyncControl
from tests_fabric.helpers.models import RandomDataset
from tests_fabric.helpers.runif import RunIf

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.fsdp.wrap import always_wrap_policy
    from torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel
else:
    xm = None
    always_wrap_policy = None
    XlaFullyShardedDataParallel = None


@RunIf(tpu=True)
@mock.patch("lightning.fabric.strategies.xla_fsdp.XLAFSDPStrategy.root_device")
def test_xla_fsdp_mp_device_dataloader_attribute(_, monkeypatch):
    dataset = RandomDataset(32, 64)
    dataloader = DataLoader(dataset)
    strategy = XLAFSDPStrategy()
    isinstance_return = True

    import torch_xla.distributed.parallel_loader as parallel_loader

    class MpDeviceLoaderMock(MagicMock):
        def __instancecheck__(self, instance):
            # to make `isinstance(dataloader, MpDeviceLoader)` pass with a mock as class
            return isinstance_return

    mp_loader_mock = MpDeviceLoaderMock()
    monkeypatch.setattr(parallel_loader, "MpDeviceLoader", mp_loader_mock)

    processed_dataloader = strategy.process_dataloader(dataloader)
    assert processed_dataloader is dataloader
    mp_loader_mock.assert_not_called()  # no-op

    isinstance_return = False
    processed_dataloader = strategy.process_dataloader(dataloader)
    mp_loader_mock.assert_called_with(dataloader, strategy.root_device)
    assert processed_dataloader.dataset == processed_dataloader._loader.dataset
    assert processed_dataloader.batch_sampler == processed_dataloader._loader.batch_sampler


@RunIf(tpu=True)
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
            with pytest.raises(ValueError, match="The optimizer does not seem to reference any XLA FSDP parameter"):
                strategy.setup_optimizer(bad_optimizer_2)


@RunIf(tpu=True)
def test_xla_fsdp_no_backward_sync():
    """Test that the backward sync control calls `.no_sync()`, and only on a module wrapped in
    XlaFullyShardedDataParallel."""

    strategy = XLAFSDPStrategy()
    assert isinstance(strategy._backward_sync_control, _XLAFSDPBackwardSyncControl)

    with pytest.raises(
        TypeError, match="is only possible if the module passed to .* is wrapped in `XlaFullyShardedDataParallel`"
    ), strategy._backward_sync_control.no_backward_sync(object()):
        pass

    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

    module = MagicMock(spec=XlaFullyShardedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module):
        pass

    module.no_sync.assert_called_once()


@RunIf(tpu=True)
def test_xla_fsdp_grad_clipping_value_error():
    strategy = XLAFSDPStrategy()
    with pytest.raises(NotImplementedError, match="does not support to clip gradients by value"):
        strategy.clip_gradients_value(Mock(), Mock(), Mock())


@RunIf(tpu=True)
def test_xla_fsdp_grad_clipping_norm_error():
    strategy = XLAFSDPStrategy()
    with pytest.raises(
        TypeError,
        match="only possible if the module.*is wrapped in `XLAFullyShardedDataParallel`",
    ):
        strategy.clip_gradients_norm(Mock(), Mock(), Mock())


def xla_fsdp_train_save_load(fabric: Fabric, tmp_path, state_dict_type):
    """Fabric launch function for test_xla_fsdp_train_save_load."""
    tmp_path = str(tmp_path)
    device = xm.xla_device()
    model_1 = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    model_1.to(device)
    model_1 = fabric.setup_module(model_1)

    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.1)
    optimizer_1 = fabric.setup_optimizers(optimizer_1)

    dataloader = DataLoader(RandomDataset(32, 64))
    dataloader = fabric.setup_dataloaders(dataloader)

    def step(model, batch):
        output = model(batch)
        return torch.nn.functional.mse_loss(output, torch.ones_like(output))

    model_1.train()
    data_iter = iter(dataloader)
    batch = next(data_iter)
    loss = step(model_1, batch)
    fabric.backward(loss)
    optimizer_1.step()
    optimizer_1.zero_grad()
    xm.mark_step()

    checkpoint_path = fabric.broadcast(tmp_path)
    checkpoint_filename = fabric.broadcast(f"{tmp_path}/fsdp-checkpoint")
    params_before = deepcopy(list(model_1.parameters()))

    state = {
        "model": model_1,
        "shard_metadata": model_1._forward_module.get_shard_metadata(),
        "optimizer": optimizer_1,  # not needed in ckpt consolidation
        "step_count": 1,
    }

    fabric.save(checkpoint_filename, state)

    world_size = xm.xrt_world_size()

    if state_dict_type == "sharded":
        expected_files = set([f"fsdp-checkpoint_rank-{i:08d}-of-{world_size:08d}.pth" for i in range(world_size)])
        assert set(os.listdir(checkpoint_path)) == expected_files

        # define a second set of model and optimizer
        model_2 = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        model_2.to(device)
        model_2 = fabric.setup_module(model_2)

        optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)
        optimizer_2 = fabric.setup_optimizers(optimizer_2)

        # load sharded checkpoints into the second set of model and optimizer
        state = {
            "model": model_2,
            "shard_metadata": model_2._forward_module.get_shard_metadata(),
            "optimizer": optimizer_2,
            "step_count": 0,
        }
        metadata = fabric.load(checkpoint_filename, state)

        # check user data in loaded state
        assert not metadata
        assert state["step_count"] == 1

        # check correctness with loaded state
        for p0, p1 in zip(params_before, model_2.parameters()):
            torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)

        # attempt to load a key not in the metadata checkpoint
        state = {"model": model_2, "coconut": 11}
        with pytest.raises(KeyError, match="The requested state contains a key 'coconut' that does not exist"):
            fabric.load(checkpoint_filename, state)

        # `strict=False` ignores the missing key
        state = {"model": model_2, "coconut": 11}
        fabric.load(checkpoint_filename, state, strict=False)
        assert state["coconut"] == 11

    if state_dict_type == "full":
        expected_files = set([f"fsdp-checkpoint_rank-{i:08d}-of-{world_size:08d}.pth" for i in range(world_size)])
        expected_files.add("fsdp-checkpoint_consolidated.pth")
        assert set(os.listdir(checkpoint_path)) == expected_files

        # define a second set of model and optimizer
        model_2 = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
        model_2.to(device)

        # load sharded checkpoints into the second model
        state = {"model": model_2}
        fabric.load(checkpoint_filename, state)

        # check that loaded state is different
        with pytest.raises(AssertionError, match="do not match"):
            for p0, p1 in zip(model_1.parameters(), model_2.parameters()):
                torch.testing.assert_close(p0, p1, atol=0, rtol=0, equal_nan=True)


@RunIf(tpu=True, standalone=True)
@pytest.mark.parametrize("state_dict_type", ["sharded", "full"])
@pytest.mark.parametrize("use_auto_wrap_policy", [True, False])
def test_xla_fsdp_train_save_load(tmp_path, state_dict_type, use_auto_wrap_policy):
    """Test XLAFSDP training, saving and loading checkpoint (both full and sharded)."""
    strategy = XLAFSDPStrategy(
        auto_wrap_policy=always_wrap_policy if use_auto_wrap_policy else None, state_dict_type=state_dict_type
    )
    fabric = Fabric(accelerator="tpu", strategy=strategy)

    fabric.launch(xla_fsdp_train_save_load, tmp_path, state_dict_type)


@RunIf(tpu=True)
def test_xla_fsdp_activation_checkpointing_setup():
    """Test XLAFSDP activation checkpointing setup."""
    from torch_xla.distributed.fsdp import checkpoint_module

    auto_wrapper_callable = lambda m, *args, **kwargs: XlaFullyShardedDataParallel(
        checkpoint_module(m), *args, **kwargs
    )
    strategy = XLAFSDPStrategy(auto_wrapper_callable=auto_wrapper_callable)

    assert auto_wrapper_callable in strategy._fsdp_kwargs.values()


def xla_fsdp_rewrap_warning(fabric: Fabric):
    """Fabric launch function for test_xla_fsdp_rewrap_warning."""
    device = xm.xla_device()
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1), torch.nn.ReLU(), XlaFullyShardedDataParallel(torch.nn.Linear(1, 1))
    )
    model.to(device)
    if xm.is_master_ordinal(local=False):
        with pytest.warns(match="the model is already wrapped"):
            model = fabric.setup_module(model)
    else:
        model = fabric.setup_module(model)
    xm.rendezvous("warning_check")
    assert not isinstance(model._forward_module, XlaFullyShardedDataParallel)
    assert isinstance(model._forward_module[2], XlaFullyShardedDataParallel)


@RunIf(min_torch="1.12", tpu=True)
def test_xla_fsdp_rewrap_warning():
    """Test XLAFSDP rewrap warning."""
    strategy = XLAFSDPStrategy(auto_wrap_policy={torch.nn.Linear})
    fabric = Fabric(accelerator="tpu", strategy=strategy)
    fabric.launch(xla_fsdp_rewrap_warning)
