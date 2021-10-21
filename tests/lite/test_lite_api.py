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
from copy import deepcopy
from unittest import mock

import pytest
import torch
import torch.distributed
import torch.nn.functional
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import RandomDataset
from tests.helpers.runif import RunIf


class BoringModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2, bias=False)

    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))


def configure_optimizers(module: nn.Module):
    return torch.optim.SGD(module.parameters(), lr=0.0001)


def configure_optimizers_schedulers(module: nn.Module):
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    return [optimizer], [lr_scheduler]


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


def test_lightning_lite_track_model_setup():
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


def test_lightning_lite_deepspeed_backward():
    with mock.patch("pytorch_lightning.plugins.DeepSpeedPlugin.setup_distributed", lambda x: x):

        class LiteRunner(LightningLite):
            def run(self):
                def fn(*args):
                    return args

                self._strategy._setup_model_and_optimizer = fn
                model = BoringModel()
                optimizer = configure_optimizers(model)
                self.setup(model, optimizer)

                model = BoringModel()
                optimizer = configure_optimizers(model)
                self.setup(model, optimizer)

                x = model(torch.randn(1, 32))
                loss = x.sum()
                self.backward(loss)

        with pytest.raises(MisconfigurationException, match="please provide the model used to perform"):
            runner = LiteRunner(strategy="deepspeed")
            runner.run()


@RunIf(min_gpus=2, deepspeed=True, special=True)
def test_deepspeed_multiple_models():
    class LiteRunner(LightningLite):
        def run(self):
            seed_everything(42)
            model = BoringModel()
            optimizer = configure_optimizers(model)
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

            seed_everything(42)
            model_1 = BoringModel()
            optimizer_1 = configure_optimizers(model_1)

            seed_everything(42)
            model_2 = BoringModel()
            optimizer_2 = configure_optimizers(model_2)

            for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
                assert torch.equal(mw_1, mw_2)

            model_1, optimizer_1 = self.setup(model_1, optimizer_1)
            model_2, optimizer_2 = self.setup(model_2, optimizer_2)

            seed_everything(42)
            for _ in range(2):
                optimizer_1.zero_grad()
                x = model_1(torch.randn(1, 32).to(self.device))
                loss = x.sum()
                self.backward(loss, model=model_1)
                optimizer_1.step()

            for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
                assert not torch.equal(mw_1, mw_2)

            seed_everything(42)
            for _ in range(2):
                optimizer_2.zero_grad()
                x = model_2(torch.randn(1, 32).to(self.device))
                loss = x.sum()
                self.backward(loss, model=model_2)
                optimizer_2.step()

            for mw_1, mw_2 in zip(model_1.state_dict().values(), model_2.state_dict().values()):
                assert torch.equal(mw_1, mw_2)

    LiteRunner(strategy=DeepSpeedPlugin(stage=3), devices=2, accelerator="gpu").run()
