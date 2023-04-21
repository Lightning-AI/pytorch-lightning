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
from functools import partial
from unittest import mock
from unittest.mock import ANY, MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import torch_xla.core.xla_model as xm

from lightning.fabric.accelerators import TPUAccelerator
from lightning.fabric.strategies import XLAFSDPStrategy
from lightning.fabric.strategies.xla_fsdp import _XLAFSDPBackwardSyncControl
from lightning.fabric.strategies.launchers.xla import _XLALauncher
from lightning.fabric.utilities.distributed import ReduceOp
from tests_fabric.helpers.models import RandomDataset
from tests_fabric.helpers.runif import RunIf
from tests_fabric.strategies.test_single_device import _MyFabricGradNorm

from torch_xla.distributed.fsdp.xla_fully_sharded_data_parallel import XlaFullyShardedDataParallel



def wrap_launch_function(fn, strategy, *args, **kwargs):
    # the launcher does not manage this automatically. explanation available in:
    # https://github.com/Lightning-AI/lightning/pull/14926#discussion_r982976718
    strategy.setup_environment()
    return fn(*args, **kwargs)


def xla_launch(fn):
    # TODO: the accelerator should be optional to just launch processes, but this requires lazy initialization
    accelerator = TPUAccelerator()
    strategy = XLAFSDPStrategy(
        accelerator=accelerator,
        parallel_devices=TPUAccelerator.get_parallel_devices(TPUAccelerator.auto_device_count()),
    )
    launcher = _XLALauncher(strategy=strategy)
    wrapped = partial(wrap_launch_function, fn, strategy)
    return launcher.launch(wrapped, strategy)


def broadcast_on_tpu_fn(strategy):
    # test broadcasting a tensor
    obj = torch.tensor(strategy.local_rank)
    # In PjRT, the local rank and global rank have no solid relation.
    # global rank may not even be contiguous on a host, because it depends on the 3D mesh structure that is formed by
    # the TPUs on all hosts in a pod. So checking a different src is not reliable
    # https://github.com/pytorch/xla/blob/v2.0.0/torch_xla/experimental/pjrt.py#L161-L163
    src = 0
    result = strategy.broadcast(obj, src)
    assert result.item() == src
    assert result.device.type == "xla"

    # test broadcasting an arbitrary object
    obj = ("ver_0.5", "logger_name", strategy.local_rank)
    result = strategy.broadcast(obj, src=src)
    assert result == ("ver_0.5", "logger_name", src)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_broadcast_on_tpu():
    """Checks if an object from the main process is broadcast to other processes correctly."""
    xla_launch(broadcast_on_tpu_fn)


def tpu_reduce_fn(strategy):
    with pytest.raises(ValueError, match="XLAFSDPStrategy only supports"):
        strategy.all_reduce(1, reduce_op="undefined")

    with pytest.raises(ValueError, match="XLAFSDPStrategy only supports"):
        strategy.all_reduce(1, reduce_op=ReduceOp.MAX)

        # it is faster to loop over here than to parameterize the test
        for reduce_op in ("mean", "AVG", "sum", ReduceOp.SUM):
            result = strategy.all_reduce(1, reduce_op=reduce_op)
            if isinstance(reduce_op, str) and reduce_op.lower() in ("mean", "avg"):
                assert result.item() == 1
            else:
                assert result.item() == 8


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_reduce():
    """Test tpu spawn all_reduce operation."""
    xla_launch(tpu_reduce_fn)


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
def tpu_all_gather_fn(strategy):
    with pytest.raises(NotImplementedError, match="only implemented for tensors"):
        strategy.all_gather([1])

    device_count = strategy.accelerator.auto_device_count()
    for sync_grads in (True, False):
        tensor = torch.tensor(1.0, requires_grad=True)
        result = strategy.all_gather(tensor, sync_grads=sync_grads)
        summed = result.sum()
        assert summed.device.type == "xla"
        assert torch.equal(summed, torch.tensor(device_count, dtype=torch.float32))
        summed.backward()
        if sync_grads:
            assert torch.equal(tensor.grad, torch.tensor(1.0))
        else:
            # As gradients are not synced, the original tensor will not have gradients.
            assert tensor.grad is None


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_all_gather():
    """Test the all_gather operation on TPU."""
    xla_launch(tpu_all_gather_fn)


@RunIf(min_torch="1.12")
@pytest.mark.parametrize("torch_ge_2_0", [False, True])
def test_xla_fsdp_setup_optimizer_validation(torch_ge_2_0):
    """Test that `setup_optimizer()` validates the param groups and reference to FSDP parameters."""
    module = nn.Linear(2, 2)
    strategy = XLAFSDPStrategy(parallel_devices=TPUAccelerator.get_parallel_devices(TPUAccelerator.auto_device_count()),)

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
    ):
        with strategy._backward_sync_control.no_backward_sync(Mock()):
            pass

    module = MagicMock(spec=XlaFullyShardedDataParallel)
    with strategy._backward_sync_control.no_backward_sync(module):
        pass

    module.no_sync.assert_called_once()


@RunIf(tpu=True)
def test_xla_fsdp_grad_clipping_value_error():
    strategy = XLAFSDPStrategy()
    with pytest.raises(
        NotImplementedError,
        match=(
            "XLAFSDP currently does not support to clip gradients by value. "
            "Consider clipping by norm instead or choose another strategy!"
        ),
    ):
        strategy.clip_gradients_value(Mock(), Mock(), Mock())
