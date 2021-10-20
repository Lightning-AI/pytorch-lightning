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
import pytest
import torch
import torch.distributed
import torch.nn.functional
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning.lite import LightningLite
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import RandomDataset


class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2, bias=False)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))


def configure_optimizers(module: nn.Module):
    return torch.optim.SGD(module.parameters(), lr=0.0001)


def test_lightning_lite_setup():
    class LiteRunner(LightningLite):
        def run(self, pass_model: bool = True):
            model = BoringModel()
            optimizer = configure_optimizers(model)
            model_lite, optimizer_lite = self.setup(model, optimizer)
            if pass_model:
                self.setup(model_lite, optimizer)
            else:
                self.setup(model, optimizer_lite)

    with pytest.raises(MisconfigurationException, match="A module should be passed only once to the"):
        runner = LiteRunner()
        runner.run()

    with pytest.raises(MisconfigurationException, match="An optimizer should be passed only once to the"):
        runner = LiteRunner()
        runner.run(pass_model=False)


def test_lightning_lite_setup_dataloaders():
    class LiteRunner(LightningLite):
        def run(self):

            dataloader = DataLoader(RandomDataset(32, 64))
            dataloader_lite = self.setup_dataloaders(dataloader)
            dataloader_lite = self.setup_dataloaders(dataloader_lite)

    with pytest.raises(MisconfigurationException, match="A dataloader should be passed only once to the"):
        runner = LiteRunner()
        runner.run()


def test_lightning_lite_track_model_with_deepspeed():
    class LiteRunner(LightningLite):
        def run(self):
            model = BoringModel()
            optimizer = configure_optimizers(model)
            self.setup(model, optimizer)
            assert not self._is_using_multiple_models

            model = BoringModel()
            optimizer = configure_optimizers(model)
            self.setup(model, optimizer)
            assert self._is_using_multiple_models

    runner = LiteRunner()
    runner.run()
