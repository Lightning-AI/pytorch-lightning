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
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from torch.nn import DataParallel

from pytorch_lightning import LightningModule
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.data_parallel import (
    LightningParallelModule,
    python_scalar_to_tensor,
    unsqueeze_scalar_tensor,
)
from pytorch_lightning.trainer.states import RunningStage
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


@pytest.mark.parametrize("wrapper_class", [
    LightningParallelModule,
    LightningDistributedModule,
])
@pytest.mark.parametrize(
    "stage", [
        ("training", "training_step"),
        ("testing", "test_step"),
        ("validating", "validation_step"),
        ("predicting", "predict_step"),
    ]
)
def test_lightning_wrapper_module_methods(wrapper_class, stage):
    """ Test that the LightningWrapper redirects .forward() to the LightningModule methods. """
    pl_module = MagicMock()
    wrapped_module = wrapper_class(pl_module)

    batch = torch.rand(5)
    batch_idx = 3

    prop, step = stage
    pl_module.trainer.sanity_checking = False

    for p in ("training", "testing", "validating", "predicting"):
        setattr(pl_module.trainer, p, p == prop)

    wrapped_module(batch, batch_idx)
    getattr(pl_module, step).assert_called_with(batch, batch_idx)


@pytest.mark.parametrize(
    "inp,expected", [
        [torch.tensor(1.0), torch.tensor([1.0])],
        [torch.tensor([2.0]), torch.tensor([2.0])],
        [torch.ones(3, 4, 5), torch.ones(3, 4, 5)],
    ]
)
def test_unsqueeze_scalar_tensor(inp, expected):
    """ Test that the utility function unsqueezes only scalar tensors. """
    assert torch.all(unsqueeze_scalar_tensor(inp).eq(expected))


@RunIf(min_gpus=2)
def test_lightning_parallel_module_unsqueeze_scalar():
    """ Test that LightningParallelModule takes care of un-squeezeing 0-dim tensors. """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            loss = output["loss"]
            loss = loss.squeeze()
            assert loss.dim() == 0
            # PyTorch usually warns about 0-dim tensors returned in DP
            return {"loss": loss}

    model = TestModel()
    model.trainer = Mock()
    model.trainer.state.stage = RunningStage.TRAINING
    batch = torch.rand(2, 32).cuda()
    batch_idx = 0

    wrapped_model = LightningParallelModule(model).cuda()
    dp_module = DataParallel(wrapped_model, device_ids=[0, 1])

    output = wrapped_model(batch, batch_idx)
    assert output["loss"].dim() == 1

    with pytest.warns(None) as record:
        output = dp_module(batch, batch_idx)

    assert output["loss"].dim() == 1
    assert not record


@pytest.mark.parametrize(
    "inp,expected", [
        [1.0, torch.tensor([1.0])],
        [2, torch.tensor([2.0])],
        [True, torch.tensor([True])],
    ]
)
def test_python_scalar_to_tensor(inp, expected):
    assert torch.all(python_scalar_to_tensor(inp).eq(expected))


@RunIf(min_gpus=1)
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda", 0)])
def test_lightning_parallel_module_python_scalar_conversion(device):
    """ Test that LightningParallelModule can convert Python scalars to tensors. """

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            # PyTorch DP does not support Python scalars, Lightning converts them to tensors
            output.update({"python scalar": 12.3})
            return output

    model = TestModel().to(device)
    model.trainer = Mock()
    model.trainer.state.stage = RunningStage.TRAINING
    batch = torch.rand(2, 32).to(device)
    batch_idx = 0

    wrapped_model = LightningParallelModule(model)
    output = wrapped_model(batch, batch_idx)
    assert output["python scalar"] == torch.tensor([12.3], device=device)


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    "nest, unnest", [
        (lambda x: x, lambda x: x),
        (lambda x: dict(data=x), lambda x: x["data"]),
        (lambda x: [x, (x, x)], lambda x: x[1][0]),
    ]
)
def test_lightning_parallel_module_device_access(nest, unnest):
    """ Test that self.device returns the correct value in replicas of DataParallel. """

    class DeviceAccessModel(LightningModule):

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 3)

        @auto_move_data
        def training_step(self, batch, batch_idx):
            batch = unnest(batch)
            assert batch.shape == torch.Size([1, 1])
            assert self.device.index == batch.item()
            assert self.device == self.layer.weight.device
            return torch.tensor(1, device=self.device)

    pl_module = DeviceAccessModel()
    # required for redirecting the forward call to training_step
    pl_module.trainer = Mock()
    pl_module.trainer.state.stage = RunningStage.TRAINING

    root_device = torch.device("cuda", 0)
    wrapped_module = LightningParallelModule(pl_module).to(root_device)
    model = DataParallel(wrapped_module, device_ids=[0, 1])

    data = torch.tensor([0.0, 1.0], device=root_device).view(2, 1)  # one value per gpu
    data = data.to(root_device)
    data = nest(data)
    output = model(data, 0)
    assert output.device == root_device
    assert pl_module.device == root_device
    assert torch.all(output.cpu().eq(torch.tensor([1, 1])))


@RunIf(min_gpus=2)
def test_lightning_parallel_module_device_access_warning():
    """ Test that we show a warning when the device can't be inferred from the input. """

    class DeviceAccessModel(LightningModule):

        def training_step(self, batch, batch_idx):
            pass

    pl_module = DeviceAccessModel()
    # required for redirecting the forward call to training_step
    pl_module.trainer = Mock()
    pl_module.trainer.state.stage = RunningStage.TRAINING

    wrapped_module = LightningParallelModule(pl_module).cuda()
    model = DataParallel(wrapped_module, device_ids=[0, 1])

    data = dict(x=1)  # contains no tensors
    with pytest.warns(UserWarning, match="Could not determine on which device the inputs are."):
        _ = model(data, 0)
