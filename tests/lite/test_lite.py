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
from unittest.mock import Mock, PropertyMock

import pytest
import torch
from torch.utils.data import DataLoader, Sampler, DistributedSampler

from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.lite.wrappers import _LiteDataLoader
from pytorch_lightning.plugins import TrainingTypePlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.runif import RunIf


class EmptyLite(LightningLite):
    def run(self):
        pass


@pytest.mark.parametrize("accelerator", ["coconut", Mock(spec=Accelerator)])
def test_unsupported_accelerator(accelerator):
    with pytest.raises(MisconfigurationException, match=f"`accelerator={repr(accelerator)}` is not a valid choice"):
        EmptyLite(accelerator=accelerator)


@pytest.mark.parametrize("strategy", ["coconut", Mock(spec=TrainingTypePlugin)])
def test_unsupported_strategy(strategy):
    with pytest.raises(MisconfigurationException, match=f"`strategy={repr(strategy)}` is not a valid choice"):
        EmptyLite(strategy=strategy)


def test_setup_dataloaders_return_type():
    """Test that the setup method returns the dataloaders wrapped as LiteDataLoader and in the right order."""
    lite = EmptyLite()

    # single dataloader
    lite_dataloader = lite.setup_dataloaders(DataLoader(range(2)))
    assert isinstance(lite_dataloader, _LiteDataLoader)

    # multiple dataloaders
    dataset0 = Mock()
    dataset1 = Mock()
    dataloader0 = DataLoader(dataset0)
    dataloader1 = DataLoader(dataset1)
    lite_dataloader0, lite_dataloader1 = lite.setup_dataloaders(dataloader0, dataloader1)
    assert isinstance(lite_dataloader0, _LiteDataLoader)
    assert isinstance(lite_dataloader1, _LiteDataLoader)
    assert lite_dataloader0.dataset is dataset0
    assert lite_dataloader1.dataset is dataset1


@mock.patch(
    "pytorch_lightning.lite.lite.LightningLite.device",
    new_callable=PropertyMock,
    return_value=torch.device("cuda", 1),
)
def test_setup_dataloaders_move_to_device(lite_device_mock):
    """Test that the setup configures LiteDataLoader to move the data to the device automatically."""
    lite = EmptyLite()
    lite_dataloaders = lite.setup_dataloaders(DataLoader(Mock()), DataLoader(Mock()), move_to_device=False)
    assert all(dl.device is None for dl in lite_dataloaders)
    lite_device_mock.assert_not_called()

    lite = EmptyLite()
    lite_dataloaders = lite.setup_dataloaders(DataLoader(Mock()), DataLoader(Mock()), move_to_device=True)
    assert all(dl.device == torch.device("cuda", 1) for dl in lite_dataloaders)
    lite_device_mock.assert_called()


def test_setup_dataloaders_distributed_sampler_not_needed():
    """Test that replace_sampler option has no effect when no distributed sampler is needed."""
    custom_sampler = Mock(spec=Sampler)
    dataloader = DataLoader(Mock(), sampler=custom_sampler)

    # keep the custom sampler when not needed to replace
    lite = EmptyLite()
    lite_dataloader = lite.setup_dataloaders(dataloader, replace_sampler=True)
    assert lite_dataloader.sampler is custom_sampler


@pytest.mark.parametrize("strategy", LightningLite._supported_strategy_types())
def test_setup_dataloaders_replace_custom_sampler(strategy):
    """Test that asking to replace a custom sampler results in an error when a distributed sampler would be needed."""
    custom_sampler = Mock(spec=Sampler)
    dataloader = DataLoader(Mock(), sampler=custom_sampler)

    # explicitly asking to replace when a custom sampler is already configured raises an exception
    lite = EmptyLite(accelerator="cpu", strategy=strategy, devices=2)
    if lite._accelerator_connector.is_distributed:
        with pytest.raises(MisconfigurationException, match="You seem to have configured a sampler in your DataLoader"):
            lite.setup_dataloaders(dataloader, replace_sampler=True)

    # setting `replace_sampler=False` leaves the sampler untouched
    lite_dataloader = lite.setup_dataloaders(dataloader, replace_sampler=False)
    assert lite_dataloader.sampler is custom_sampler


@pytest.mark.parametrize("strategy", LightningLite._supported_strategy_types())
@pytest.mark.parametrize("shuffle", [True, False])
def test_setup_dataloaders_replace_standard_sampler(shuffle, strategy):
    """Test that Lite replaces the default samplers with DistributedSampler automatically."""
    lite = EmptyLite(accelerator="cpu", strategy=strategy, devices=2)
    is_distributed = lite._accelerator_connector.is_distributed
    lite_dataloader = lite.setup_dataloaders(DataLoader(range(3), shuffle=shuffle))
    assert not is_distributed or isinstance(lite_dataloader.sampler, DistributedSampler)


@pytest.mark.parametrize(
    "accelerator, expected",
    [
        ("cpu", torch.device("cpu")),
        pytest.param("gpu", torch.device("cuda", 0), marks=RunIf(min_gpus=1)),
        pytest.param("tpu", torch.device("xla", 0), marks=RunIf(tpu=True)),
    ],
)
def test_to_device(accelerator, expected):
    lite = EmptyLite(accelerator=accelerator, devices=1)

    module = torch.nn.Linear(2, 3)
    module = lite.to_device(module)
    assert all(param.device == expected for param in module.parameters())

    tensor = torch.rand(2, 2)
    tensor = lite.to_device(tensor)
    assert tensor.device == expected

    collection = {"data": torch.rand(2, 2)}
    collection = lite.to_device(collection)
    assert collection["data"].device == expected
