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
from copy import deepcopy
from unittest import mock
from unittest.mock import MagicMock, Mock, PropertyMock

import pytest
import torch
import torch.distributed
import torch.nn.functional
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from pytorch_lightning.lite import LightningLite
from pytorch_lightning.lite.wrappers import (
    _LiteDataLoader,
    _LiteModule,
    _LiteOptimizer,
    _replace_dataloader_init_method,
)
from pytorch_lightning.plugins import DeepSpeedPlugin, PrecisionPlugin, TrainingTypePlugin
from pytorch_lightning.utilities import DistributedType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import pl_worker_init_function
from tests.helpers.runif import RunIf


class EmptyLite(LightningLite):
    def run(self):
        pass


class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2, bias=False)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))


def test_unsupported_accelerator():
    accelerator = "coconut"
    with pytest.raises(MisconfigurationException, match=f"`accelerator={repr(accelerator)}` is not a valid choice"):
        EmptyLite(accelerator=accelerator)


def test_unsupported_strategy():
    strategy = "coconut"
    with pytest.raises(MisconfigurationException, match=f"`strategy={repr(strategy)}` is not a valid choice"):
        EmptyLite(strategy=strategy)


def test_run_input_output():
    """Test that the dynamically patched run() method receives the input arguments and returns the result."""

    class Lite(LightningLite):

        run_args = ()
        run_kwargs = {}

        def run(self, *args, **kwargs):
            self.run_args = args
            self.run_kwargs = kwargs
            return "result"

    lite = Lite()
    result = lite.run(1, 2, three=3)
    assert result == "result"
    assert lite.run_args == (1, 2)
    assert lite.run_kwargs == {"three": 3}


def test_setup_optimizers():
    """Test that setup_optimizers can handle no optimizers, one optimizer, or multiple optimizers."""
    lite = EmptyLite()
    model = nn.Linear(1, 2)
    optimizer0 = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.1)

    # no optimizer
    lite_model = lite.setup(model)
    assert isinstance(lite_model, _LiteModule)
    assert lite_model.module is model

    # single optimizer
    lite_model, lite_optimizer = lite.setup(model, optimizer0)
    assert isinstance(lite_model, _LiteModule)
    assert isinstance(lite_optimizer, _LiteOptimizer)
    assert lite_model.module is model
    assert lite_optimizer.optimizer is optimizer0

    # multiple optimizers
    lite_model, lite_optimizer0, lite_optimizer1 = lite.setup(model, optimizer0, optimizer1)
    assert isinstance(lite_model, _LiteModule)
    assert isinstance(lite_optimizer0, _LiteOptimizer)
    assert isinstance(lite_optimizer1, _LiteOptimizer)
    assert lite_model.module is model
    assert lite_optimizer0.optimizer is optimizer0
    assert lite_optimizer1.optimizer is optimizer1


def test_setup_twice_fails():
    """Test that calling setup with a model or optimizer that is already wrapped fails."""
    lite = EmptyLite()
    model = nn.Linear(1, 2)
    optimizer = torch.optim.Adam(model.parameters())

    lite_model, lite_optimizer = lite.setup(model, optimizer)
    with pytest.raises(MisconfigurationException, match="A model should be passed only once to the"):
        lite.setup(lite_model, optimizer)

    lite_model, lite_optimizer = lite.setup(model, optimizer)
    with pytest.raises(MisconfigurationException, match="An optimizer should be passed only once to the"):
        lite.setup(model, lite_optimizer)


def test_setup_tracks_num_models():
    """Test that setup() tracks how many times it has setup a model."""
    lite = EmptyLite()
    model = nn.Linear(1, 2)
    optimizer = torch.optim.Adam(model.parameters())

    assert lite._models_setup == 0
    lite.setup(model, optimizer)
    assert lite._models_setup == 1

    lite.setup(model, optimizer)
    assert lite._models_setup == 2


def test_setup_dataloaders_unsupported_type():
    """Test that the setup_dataloaders method fails when provided with non-DataLoader objects."""
    lite = EmptyLite()
    with pytest.raises(MisconfigurationException, match="Only PyTorch DataLoader are currently supported"):
        lite.setup_dataloaders(range(2))  # type: ignore


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


def test_setup_dataloaders_with_custom_type():
    """Test that Lite intercepts arguments passed to custom subclasses of torch.utils.DataLoader and sets them as
    attributes."""

    class DataLoaderSubclass1(DataLoader):
        def __init__(self, attribute1, *args, **kwargs):
            # intentionally not setting this attribute, calling super with different args
            # self.attribute1 = attribute1
            super().__init__(*args, **kwargs)

    class DataLoaderSubclass2(DataLoaderSubclass1):
        def __init__(self, attribute1, attribute2, *args, **kwargs):
            # intentionally not setting this attribute, calling super with different args
            # self.attribute2 = attribute2
            super().__init__(attribute1, *args, **kwargs)

    class LiteWithCustomDataLoader(LightningLite):
        def run(self):
            dataloader = DataLoaderSubclass2("attribute1", "attribute2", dataset=range(4), batch_size=2)
            assert dataloader.attribute1 == "attribute1"
            assert dataloader.attribute2 == "attribute2"
            lite_dataloader = self.setup_dataloaders(dataloader)
            assert lite_dataloader.attribute1 == "attribute1"
            assert lite_dataloader.attribute2 == "attribute2"

    LiteWithCustomDataLoader().run()


def test_setup_dataloaders_twice_fails():
    """Test that calling setup_dataloaders with a dataloader that is already wrapped fails."""
    lite = EmptyLite()
    dataloader = DataLoader(range(2))
    lite_dataloader = lite.setup_dataloaders(dataloader)

    with pytest.raises(MisconfigurationException, match="A dataloader should be passed only once to the"):
        lite.setup_dataloaders(lite_dataloader)


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


@mock.patch.dict(os.environ, {}, clear=True)
def test_seed_everything():
    """Test that seed everything is static and sets the worker init function on the dataloader."""
    EmptyLite.seed_everything(3)

    lite = EmptyLite()
    lite_dataloader = lite.setup_dataloaders(DataLoader(Mock()))

    assert lite_dataloader.worker_init_fn.func is pl_worker_init_function
    assert os.environ == {"PL_GLOBAL_SEED": "3", "PL_SEED_WORKERS": "1"}


@pytest.mark.parametrize(
    "strategy",
    [
        DistributedType.DP,
        DistributedType.DDP,
        DistributedType.DDP_SPAWN,
        pytest.param(DistributedType.DEEPSPEED, marks=RunIf(deepspeed=True)),
        pytest.param(DistributedType.DDP_SHARDED, marks=RunIf(fairscale=True)),
        pytest.param(DistributedType.DDP_SHARDED_SPAWN, marks=RunIf(fairscale=True)),
    ],
)
def test_setup_dataloaders_replace_custom_sampler(strategy):
    """Test that asking to replace a custom sampler results in an error when a distributed sampler would be
    needed."""
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


@pytest.mark.parametrize(
    "strategy",
    [
        DistributedType.DP,
        DistributedType.DDP,
        DistributedType.DDP_SPAWN,
        pytest.param(DistributedType.DEEPSPEED, marks=RunIf(deepspeed=True)),
        pytest.param(DistributedType.DDP_SHARDED, marks=RunIf(fairscale=True)),
        pytest.param(DistributedType.DDP_SHARDED_SPAWN, marks=RunIf(fairscale=True)),
    ],
)
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
    """Test that the to_device method can move various objects to the device determined by the accelerator."""
    lite = EmptyLite(accelerator=accelerator, devices=1)

    # module
    module = torch.nn.Linear(2, 3)
    module = lite.to_device(module)
    assert all(param.device == expected for param in module.parameters())

    # tensor
    tensor = torch.rand(2, 2)
    tensor = lite.to_device(tensor)
    assert tensor.device == expected

    # collection
    collection = {"data": torch.rand(2, 2), "int": 1}
    collection = lite.to_device(collection)
    assert collection["data"].device == expected


def test_rank_properties():
    """Test that the rank properties are determined by the strategy."""
    lite = EmptyLite()
    lite._strategy = Mock(spec=TrainingTypePlugin)
    lite._strategy.world_size = 1000
    assert lite.world_size == 1000
    lite._strategy.global_rank = 100
    assert lite.global_rank == 100
    lite._strategy.local_rank = 10
    assert lite.local_rank == 10
    lite._strategy.node_rank = 1
    assert lite.node_rank == 1


def test_backward():
    """Test that backward() calls into the precision plugin."""
    lite = EmptyLite()
    lite._precision_plugin = Mock(spec=PrecisionPlugin)
    loss = Mock()
    lite.backward(loss, "arg", keyword="kwarg")
    lite._precision_plugin._run_backward.assert_called_with(loss, None, "arg", keyword="kwarg")


@RunIf(deepspeed=True)
def test_backward_model_input_required():
    """Test that when using deepspeed and multiple models, backward() requires the model as input."""
    lite = EmptyLite(strategy="deepspeed")

    model0 = nn.Linear(1, 2)
    model1 = nn.Linear(1, 2)

    optimizer0 = torch.optim.Adam(model0.parameters())
    optimizer1 = torch.optim.Adam(model1.parameters())

    lite._strategy._setup_model_and_optimizer = lambda *args: args

    lite.setup(model0, optimizer0)
    lite.setup(model1, optimizer1)

    loss = model0(torch.randn(1, 1)).sum()

    with pytest.raises(MisconfigurationException, match="please provide the model used to perform"):
        lite.backward(loss)


def test_autocast():
    """Test that the Lite autocast context manager lets the precision plugin handle casting."""
    lite = EmptyLite()
    lite._precision_plugin.forward_context = MagicMock()

    lite._precision_plugin.forward_context().__enter__.assert_not_called()
    with lite.autocast():
        lite._precision_plugin.forward_context().__enter__.assert_called()
    lite._precision_plugin.forward_context().__exit__.assert_called()


@RunIf(min_gpus=2, deepspeed=True, special=True)
def test_deepspeed_multiple_models():
    class Lite(LightningLite):
        def run(self):
            model = BoringModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
            model, optimizer = self.setup(model, optimizer)
            state_dict = deepcopy(model.state_dict())

            for _ in range(2):
                optimizer.zero_grad()
                x = model(torch.randn(1, 32).to(self.device))
                loss = x.sum()
                self.backward(loss, model=model)
                optimizer.step()

            for mw_b, mw_a in zip(state_dict.values(), model.state_dict().values()):
                assert not torch.equal(mw_b, mw_a)

            self.seed_everything(42)
            model_1 = BoringModel()
            optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.0001)

            self.seed_everything(42)
            model_2 = BoringModel()
            optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.0001)

            for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
                assert torch.equal(mw_1, mw_2)

            model_1, optimizer_1 = self.setup(model_1, optimizer_1)
            model_2, optimizer_2 = self.setup(model_2, optimizer_2)

            self.seed_everything(42)
            data_list = []
            for _ in range(2):
                optimizer_1.zero_grad()
                data = torch.randn(1, 32).to(self.device)
                data_list.append(data)
                x = model_1(data)
                loss = x.sum()
                self.backward(loss, model=model_1)
                optimizer_1.step()

            for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
                assert not torch.equal(mw_1, mw_2)

            for data in data_list:
                optimizer_2.zero_grad()
                x = model_2(data)
                loss = x.sum()
                self.backward(loss, model=model_2)
                optimizer_2.step()

            for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
                assert torch.equal(mw_1, mw_2)

            # Verify collectives works as expected
            ranks = self.all_gather(torch.tensor([self.local_rank]).to(self.device))
            assert torch.equal(ranks.cpu(), torch.tensor([[0], [1]]))
            assert self.broadcast(True)
            assert self.is_global_zero == (self.local_rank == 0)

    Lite(strategy=DeepSpeedPlugin(stage=3, logging_batch_size_per_gpu=1), devices=2, accelerator="gpu").run()


def test_replace_dataloader_init_method():
    """Test that the context manager enables to save the parameters passed to the DataLoader __init__ method."""

    class CustomDataLoader(DataLoader):
        def __init__(self, extra_argument: int, *args, **kwargs):
            super().__init__(*args, **kwargs)

    dataloader = CustomDataLoader(extra_argument=1, dataset=range(1))
    lite = EmptyLite()
    with pytest.raises(MisconfigurationException, match="extra_argument"):
        dataloader = lite.setup_dataloaders(dataloader)

    with _replace_dataloader_init_method():
        dataloader = CustomDataLoader(extra_argument=1, dataset=range(1))
        assert dataloader.extra_argument == 1
        dataloader = lite.setup_dataloaders(dataloader)

        dataloader = CustomDataLoader(1, range(1))
        assert dataloader.extra_argument == 1
        dataloader = lite.setup_dataloaders(dataloader)
