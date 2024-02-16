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
from lightning.fabric.accelerators.xla import _XLA_GREATER_EQUAL_2_1, XLAAccelerator
from lightning.fabric.strategies import XLAStrategy
from lightning.fabric.strategies.launchers.xla import _XLALauncher
from lightning.fabric.utilities.distributed import ReduceOp
from lightning.fabric.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from tests_fabric.helpers.datasets import RandomDataset
from tests_fabric.helpers.runif import RunIf


def wrap_launch_function(fn, strategy, *args, **kwargs):
    # the launcher does not manage this automatically. explanation available in:
    # https://github.com/Lightning-AI/lightning/pull/14926#discussion_r982976718
    strategy.setup_environment()
    return fn(*args, **kwargs)


def xla_launch(fn, strategy=None):
    # TODO: the accelerator should be optional to just launch processes, but this requires lazy initialization
    if not strategy:
        accelerator = XLAAccelerator()
        strategy = XLAStrategy(
            accelerator=accelerator,
            parallel_devices=XLAAccelerator.get_parallel_devices(XLAAccelerator.auto_device_count()),
        )
    launcher = _XLALauncher(strategy=strategy)
    wrapped = partial(wrap_launch_function, fn, strategy)
    return launcher.launch(wrapped, strategy)


def broadcast_on_tpu_fn(strategy):
    # test broadcasting a tensor
    obj = torch.tensor(strategy.global_rank)
    assert obj.device.type == "cpu"
    # In PjRT, the local rank and global rank have no solid relation.
    # global rank may not even be contiguous on a host, because it depends on the 3D mesh structure that is formed by
    # the TPUs on all hosts in a pod. So checking a different src is not reliable
    # https://github.com/pytorch/xla/blob/v2.0.0/torch_xla/experimental/pjrt.py#L161-L163
    src = 0
    result = strategy.broadcast(obj, src)
    assert result.item() == src
    assert result.device.type == "cpu"  # the original device is preserved

    # test broadcasting an arbitrary object
    tensor = torch.tensor(strategy.global_rank, device=strategy.root_device, dtype=torch.bfloat16)
    obj = ("ver_0.5", "logger_name", strategy.global_rank, tensor)
    result = strategy.broadcast(obj, src=src)
    assert result == ("ver_0.5", "logger_name", src, ANY)
    assert result[3].device.type == "xla"  # the original device is preserved
    assert result[3].dtype == torch.bfloat16


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_broadcast_on_tpu():
    """Checks if an object from the main process is broadcast to other processes correctly."""
    xla_launch(broadcast_on_tpu_fn)


def tpu_reduce_fn(strategy):
    with pytest.raises(ValueError, match="XLAStrategy only supports"):
        strategy.all_reduce(1, reduce_op="undefined")

    with pytest.raises(ValueError, match="XLAStrategy only supports"):
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
@mock.patch("lightning.fabric.strategies.xla.XLAStrategy.root_device")
def test_xla_mp_device_dataloader_attribute(_, monkeypatch):
    dataset = RandomDataset(32, 64)
    dataloader = DataLoader(dataset)
    strategy = XLAStrategy()
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


def tpu_all_gather_fn(strategy):
    with pytest.raises(NotImplementedError, match="only implemented for tensors"):
        strategy.all_gather([1])

    for sync_grads in (True, False):
        tensor = torch.tensor(1.0, requires_grad=True)
        result = strategy.all_gather(tensor, sync_grads=sync_grads)
        summed = result.sum()
        assert summed.device.type == "cpu"  # the original device is preserved
        assert torch.equal(summed, torch.tensor(strategy.world_size, dtype=torch.float32))
        if not _XLA_GREATER_EQUAL_2_1:
            summed.backward()
        if sync_grads:
            if _XLA_GREATER_EQUAL_2_1:
                # in 2.1, sync_grads=False makes it so that you cannot call .backward even if it originally had set
                # requires_grad=True
                summed.backward()
            assert torch.equal(tensor.grad, torch.tensor(1.0))
        else:
            # As gradients are not synced, the original tensor will not have gradients.
            assert tensor.grad is None


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_all_gather():
    """Test the all_gather operation on TPU."""
    xla_launch(tpu_all_gather_fn)


def tpu_sync_module_states_fn(sync_module_states, strategy):
    seed_everything(strategy.local_rank)  # force the model to have different weights across ranks
    model = torch.nn.Linear(1, 1).to(strategy.root_device)
    model = strategy.setup_module(model)
    gathered = strategy.all_gather(model.weight)
    for t in gathered[1:]:
        if sync_module_states:
            assert torch.equal(gathered[0], t)
        else:
            assert not torch.equal(gathered[0], t)


@RunIf(tpu=True)
@pytest.mark.parametrize("sync_module_states", [True, False])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_sync_module_states(sync_module_states):
    """Test sync_module_states."""
    accelerator = XLAAccelerator()
    strategy = XLAStrategy(
        accelerator=accelerator,
        parallel_devices=XLAAccelerator.get_parallel_devices(XLAAccelerator.auto_device_count()),
        sync_module_states=sync_module_states,
    )
    partial_fn = partial(tpu_sync_module_states_fn, sync_module_states)
    xla_launch(partial_fn, strategy)


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_rank_properties_access(xla_available):
    """Test that the strategy returns the expected values depending on whether we're in the main process or not."""
    strategy = XLAStrategy()
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
