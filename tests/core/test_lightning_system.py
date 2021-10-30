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
from abc import ABC
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.core.system import LightningSystem, load_from_checkpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import RandomDataset


def test_lightning_system(tmpdir):

    system = LightningSystem()
    assert not system.has_module
    assert system.module_name is None

    linear = nn.Linear(in_features=1, out_features=1)
    system.model = linear
    assert system._args == ()
    assert system._kwargs == dict(in_features=1, out_features=1)
    assert system.has_module
    assert system.module_name == "model"

    with pytest.raises(
        MisconfigurationException,
        match="A `LightningSystem` supports only a single nn.Module expects `torchmetrics.Metric`",
    ):
        setattr(system, "model_1", nn.Linear(in_features=1, out_features=1))

    with pytest.raises(
        MisconfigurationException,
        match="A `LightningSystem` doesn't support parameters.",
    ):
        setattr(system, "weight", nn.Parameter(torch.Tensor(1, 1)))

    state_dict = deepcopy(system.state_dict())
    assert state_dict["weight"] == linear.state_dict()["weight"]

    system.load_state_dict(nn.Linear(in_features=1, out_features=1).state_dict())
    assert system.state_dict()["weight"] != state_dict["weight"]

    class DummyModel(nn.Module):
        def __init__(self, name: str, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

    class System(LightningSystem):
        def __init__(self, name: str, in_features, out_features, random):
            super().__init__()
            self.save_hyperparameters()
            self.module = DummyModel(name, in_features, out_features + 1)

    system = System("dummy", 1, 1, "random")

    p = os.path.join(tmpdir, "model.pt")
    torch.save(state_dict, p)
    linear = load_from_checkpoint(nn.Linear, p)
    assert linear.state_dict()["weight"] == state_dict["weight"]

    class AbstractGAN(ABC):

        generator: nn.Module
        discriminator: nn.Module

    class Model(AbstractGAN, nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.generator = nn.Linear(dim, 1)
            self.discriminator = nn.Linear(1, 1)

        def forward(self, x):
            return self.generator(x)

    class GANSystem(LightningSystem):
        def __init__(self, module_cls, *args, **kwargs):
            super().__init__()

            self.gan = module_cls(*args, **kwargs)
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            # Implement a GAN there.
            pass

        def configure_optimizers(self):
            opt_g = torch.optim.Adam(self.gan.generator.parameters(), lr=0.01)
            opt_d = torch.optim.Adam(self.gan.discriminator.parameters())
            return [opt_g, opt_d], []

        def train_dataloader(self):
            return DataLoader(RandomDataset(2, 2))

    system = GANSystem(Model, 5)
    trainer = Trainer(fast_dev_run=False, max_epochs=1)
    trainer.fit(system)

    state_dict = system.state_dict()
    for k, v in system.gan.state_dict().items():
        assert torch.equal(state_dict[k], v)

    torch.save(state_dict, p)
    model = load_from_checkpoint(Model, p)
    assert model.generator.weight.shape == torch.Size([1, 5])
