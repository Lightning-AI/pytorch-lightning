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
from unittest.mock import MagicMock, Mock

import pytest
import torch
from torch.utils.data import DataLoader

from lightning.fabric.accelerators import TPUAccelerator
from lightning.fabric.strategies import XLAStrategy
from lightning.fabric.strategies.launchers.xla import _XLALauncher
from lightning.fabric.utilities.distributed import ReduceOp
from tests_fabric.helpers.dataloaders import CustomNotImplementedErrorDataloader
from tests_fabric.helpers.models import RandomDataset, RandomIterableDataset
from tests_fabric.helpers.runif import RunIf


def wrap_launch_function(fn, strategy, *args, **kwargs):
    # the launcher does not manage this automatically. explanation available in:
    # https://github.com/Lightning-AI/lightning/pull/14926#discussion_r982976718
    strategy.setup_environment()
    return fn(*args, **kwargs)


def xla_launch(fn):
    # TODO: the accelerator should be optional to just launch processes, but this requires lazy initialization
    accelerator = TPUAccelerator()
    strategy = XLAStrategy(accelerator=accelerator, parallel_devices=list(range(8)))
    launcher = _XLALauncher(strategy=strategy)
    wrapped = partial(wrap_launch_function, fn, strategy)
    return launcher.launch(wrapped, strategy)


def broadcast_on_tpu_fn(strategy):
    obj = ("ver_0.5", "logger_name", strategy.local_rank)
    result = strategy.broadcast(obj)
    assert result == ("ver_0.5", "logger_name", 0)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_broadcast_on_tpu():
    """Checks if an object from the main process is broadcasted to other processes correctly."""
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


_loader = DataLoader(RandomDataset(32, 64))
_iterable_loader = DataLoader(RandomIterableDataset(32, 64))
_loader_no_len = CustomNotImplementedErrorDataloader(_loader)


@RunIf(tpu=True)
@pytest.mark.parametrize("dataloader", [None, _iterable_loader, _loader_no_len])
@mock.patch("lightning.fabric.strategies.xla.XLAStrategy.root_device")
def test_xla_validate_unsupported_iterable_dataloaders(_, dataloader, monkeypatch):
    """Test that the XLAStrategy validates against dataloaders with no length defined on datasets (iterable
    dataset)."""
    import torch_xla.distributed.parallel_loader as parallel_loader

    monkeypatch.setattr(parallel_loader, "MpDeviceLoader", Mock())

    with pytest.raises(TypeError, match="TPUs do not currently support"):
        XLAStrategy().process_dataloader(dataloader)


def tpu_all_gather_fn(strategy):
    for sync_grads in [True, False]:
        tensor = torch.tensor(1.0, device=strategy.root_device, requires_grad=True)
        result = strategy.all_gather(tensor, sync_grads=sync_grads)
        summed = result.sum()
        assert torch.equal(summed, torch.tensor(8.0))
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
