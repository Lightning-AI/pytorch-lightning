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
import os
from functools import partial
from unittest import mock
from unittest.mock import Mock

import pytest
from tests_lite.helpers.dataloaders import CustomNotImplementedErrorDataloader
from tests_lite.helpers.models import RandomDataset, RandomIterableDataset
from tests_lite.helpers.runif import RunIf
from torch.utils.data import DataLoader

from lightning_lite.strategies import XLAStrategy
from lightning_lite.strategies.launchers.xla import _XLALauncher
from lightning_lite.utilities.distributed import ReduceOp
from pytorch_lightning.accelerators import TPUAccelerator


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
        strategy.reduce(1, reduce_op="undefined")

    with pytest.raises(ValueError, match="XLAStrategy only supports"):
        strategy.reduce(1, reduce_op=ReduceOp.MAX)

        # it is faster to loop over here than to parameterize the test
        for reduce_op in ("mean", "AVG", "sum", ReduceOp.SUM):
            result = strategy.reduce(1, reduce_op=reduce_op)
            if isinstance(reduce_op, str) and reduce_op.lower() in ("mean", "avg"):
                assert result.item() == 1
            else:
                assert result.item() == 8


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_tpu_reduce():
    """Test tpu spawn reduce operation."""
    xla_launch(tpu_reduce_fn)


@RunIf(tpu=True)
@mock.patch("lightning_lite.strategies.xla.XLAStrategy.root_device")
def test_xla_mp_device_dataloader_attribute(_, monkeypatch):
    import torch_xla.distributed.parallel_loader as parallel_loader

    mp_loader_mock = Mock()
    monkeypatch.setattr(parallel_loader, "MpDeviceLoader", mp_loader_mock)

    dataset = RandomDataset(32, 64)
    dataloader = DataLoader(dataset)
    strategy = XLAStrategy()
    processed_dataloader = strategy.process_dataloader(dataloader)
    mp_loader_mock.assert_called_with(dataloader, strategy.root_device)
    assert processed_dataloader.dataset == processed_dataloader._loader.dataset


_loader = DataLoader(RandomDataset(32, 64))
_iterable_loader = DataLoader(RandomIterableDataset(32, 64))
_loader_no_len = CustomNotImplementedErrorDataloader(_loader)


@RunIf(tpu=True)
@pytest.mark.parametrize("dataloader", [None, _iterable_loader, _loader_no_len])
@mock.patch("lightning_lite.strategies.xla.XLAStrategy.root_device")
def test_xla_validate_unsupported_iterable_dataloaders(_, dataloader, monkeypatch):
    """Test that the XLAStrategy validates against dataloaders with no length defined on datasets (iterable
    dataset)."""
    import torch_xla.distributed.parallel_loader as parallel_loader

    monkeypatch.setattr(parallel_loader, "MpDeviceLoader", Mock())

    with pytest.raises(TypeError, match="TPUs do not currently support"):
        XLAStrategy().process_dataloader(dataloader)
