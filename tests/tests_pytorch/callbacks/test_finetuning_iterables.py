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
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BaseFinetuning


class _FeatureModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([1.1, 1.3]))
        self.projection = nn.Linear(2, 2)
        with torch.no_grad():
            self.projection.weight.fill_(0.2)
            self.projection.bias.fill_(0.3)

    def forward(self, inputs):
        return self.projection(inputs) * self.scale


def _head():
    head = nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.fill_(0.4)
        head.bias.fill_(0.5)
    return head


def _modules_in_form(module, form):
    if form == "module":
        return module
    if form == "list":
        return [module]
    if form == "tuple":
        return (module,)
    if form == "module_dict":
        return nn.ModuleDict({"features": module})
    if form == "module_list":
        return nn.ModuleList([module])
    if form == "generator":
        return (item for item in [module])
    if form == "iterator":
        return iter([module])
    if form == "nested_generator":
        return [(), [(item for item in [module])]]
    if form == "generator_of_generators":
        return ((item for item in group) for group in [(module,)])
    raise AssertionError(form)


@pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize(
    "form",
    [
        "module",
        "list",
        "tuple",
        "module_dict",
        "module_list",
        "generator",
        "iterator",
        "nested_generator",
        "generator_of_generators",
    ],
)
def test_unfreezing_one_shot_modules_matches_list_training(optimizer_class, form):
    features, head = _FeatureModule(), _head()
    BaseFinetuning.freeze(features)
    reference_features, reference_head = deepcopy(features), deepcopy(head)
    optimizer = optimizer_class(head.parameters(), lr=0.05)
    reference = optimizer_class(reference_head.parameters(), lr=0.05)
    before = deepcopy(features.state_dict())

    BaseFinetuning.unfreeze_and_add_param_group(_modules_in_form(features, form), optimizer, lr=0.03)
    BaseFinetuning.unfreeze_and_add_param_group([reference_features], reference, lr=0.03)

    assert all(parameter.requires_grad for parameter in features.parameters())
    assert len(optimizer.param_groups) == 2
    assert {id(parameter) for parameter in optimizer.param_groups[1]["params"]} == {
        id(parameter) for parameter in features.parameters()
    }
    assert len(optimizer.param_groups[1]["params"]) == len(list(features.parameters()))
    for step in range(3):
        inputs = torch.tensor([[1.0 + 0.1 * step, -0.5], [0.4, 0.7]])
        for feature_model, head_model, opt in (
            (features, head, optimizer),
            (reference_features, reference_head, reference),
        ):
            opt.zero_grad(set_to_none=True)
            head_model(feature_model(inputs)).square().mean().backward()
            opt.step()
        for actual, expected in zip(features.parameters(), reference_features.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(head.parameters(), reference_head.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not torch.equal(value, before[name]) for name, value in features.state_dict().items())


class _FineTuneModel(LightningModule):
    def __init__(self, optimizer_class):
        super().__init__()
        self.features = _FeatureModule()
        self.head = _head()
        self.optimizer_class = optimizer_class

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        return (self.head(self.features(inputs)) - targets).square().mean()

    def configure_optimizers(self):
        return self.optimizer_class((parameter for parameter in self.parameters() if parameter.requires_grad), lr=0.05)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(4, 2), torch.zeros(4, 1)), batch_size=2)


class _UnfreezeModules(BaseFinetuning):
    def __init__(self, form):
        super().__init__()
        self.form = form

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.features)

    def finetune_function(self, pl_module, epoch, optimizer):
        if epoch == 1:
            self.unfreeze_and_add_param_group(_modules_in_form(pl_module.features, self.form), optimizer, lr=0.03)


@pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize("form", ["list", "generator", "nested_generator"])
def test_trainer_updates_unfrozen_generator_parameters(tmp_path, optimizer_class, form):
    model, reference = _FineTuneModel(optimizer_class), _FineTuneModel(optimizer_class)
    initial = deepcopy(model.features.state_dict())
    for current, module_form, directory in ((model, form, "actual"), (reference, "list", "reference")):
        trainer = Trainer(
            default_root_dir=tmp_path / directory,
            accelerator="cpu",
            devices=1,
            max_epochs=2,
            callbacks=[_UnfreezeModules(module_form)],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(current)
        assert trainer.global_step == 4

    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, reference.state_dict()[name], rtol=0, atol=0)
    assert all(not torch.equal(value, initial[name]) for name, value in model.features.state_dict().items())


@pytest.mark.parametrize("nested", [False, True])
def test_empty_module_iterators_do_not_add_optimizer_groups(nested):
    head = _head()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.05)
    modules = iter(())
    if nested:
        modules = [modules]

    BaseFinetuning.unfreeze_and_add_param_group(modules, optimizer)

    assert len(optimizer.param_groups) == 1
