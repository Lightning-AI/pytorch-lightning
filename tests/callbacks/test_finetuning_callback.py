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
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import BackboneFinetuning, BaseFinetuning
from pytorch_lightning.callbacks.base import Callback
from tests.helpers import BoringModel, RandomDataset


def test_finetuning_callback(tmpdir):
    """Test finetuning callbacks works as expected"""

    seed_everything(42)

    class FinetuningBoringModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(32, 32, bias=False), nn.BatchNorm1d(32), nn.ReLU())
            self.layer = torch.nn.Linear(32, 2)
            self.backbone.has_been_used = False

        def training_step(self, batch, batch_idx):
            output = self(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def forward(self, x):
            self.backbone.has_been_used = True
            x = self.backbone(x)
            return self.layer(x)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
            return [optimizer], [lr_scheduler]

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

    class TestCallback(BackboneFinetuning):

        def on_train_epoch_end(self, trainer, pl_module, outputs):
            epoch = trainer.current_epoch
            if self.unfreeze_backbone_at_epoch <= epoch:
                optimizer = trainer.optimizers[0]
                current_lr = optimizer.param_groups[0]['lr']
                backbone_lr = self.previous_backbone_lr
                if epoch < 6:
                    assert backbone_lr <= current_lr
                else:
                    assert backbone_lr == current_lr

    model = FinetuningBoringModel()
    callback = TestCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    trainer = Trainer(
        limit_train_batches=1,
        default_root_dir=tmpdir,
        callbacks=[callback],
        max_epochs=8,
    )
    trainer.fit(model)

    assert model.backbone.has_been_used


def test_finetuning_callback_warning(tmpdir):
    """Test finetuning callbacks works as expected"""

    seed_everything(42)

    class FinetuningBoringModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(32, 2, bias=False)
            self.layer = None
            self.backbone.has_been_used = False

        def training_step(self, batch, batch_idx):
            output = self(batch)
            loss = self.loss(batch, output)
            return {"loss": loss}

        def forward(self, x):
            self.backbone.has_been_used = True
            x = self.backbone(x)
            return x

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=2)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
            return optimizer

    class TestCallback(BackboneFinetuning):

        def finetune_function(self, pl_module, epoch: int, optimizer, opt_idx: int):
            """Called when the epoch begins."""

            if epoch == 0:
                self.unfreeze_and_add_param_group(
                    pl_module.backbone, optimizer, 0.1, train_bn=self.train_bn, initial_denom_lr=self.initial_denom_lr
                )

    model = FinetuningBoringModel()
    model.validation_step = None
    callback = TestCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    with pytest.warns(UserWarning, match="Did you init your optimizer in"):
        trainer = Trainer(
            limit_train_batches=1,
            default_root_dir=tmpdir,
            callbacks=[callback],
            max_epochs=2,
        )
        trainer.fit(model)

    assert model.backbone.has_been_used


def test_freeze_unfreeze_function(tmpdir):
    """Test freeze properly sets requires_grad on the modules"""

    seed_everything(42)

    class FreezeModel(LightningModule):

        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 2))

    model = FreezeModel()
    BaseFinetuning.freeze(model, train_bn=True)
    assert not model.backbone[0].weight.requires_grad
    assert model.backbone[1].weight.requires_grad
    assert not model.backbone[3].weight.requires_grad

    BaseFinetuning.freeze(model, train_bn=False)
    assert not model.backbone[0].weight.requires_grad
    assert not model.backbone[1].weight.requires_grad
    assert not model.backbone[3].weight.requires_grad

    BaseFinetuning.make_trainable(model)
    assert model.backbone[0].weight.requires_grad
    assert model.backbone[1].weight.requires_grad
    assert model.backbone[3].weight.requires_grad

    BaseFinetuning.freeze(model.backbone[0], train_bn=False)
    assert not model.backbone[0].weight.requires_grad

    BaseFinetuning.freeze(([(model.backbone[1]), [model.backbone[3]]]), train_bn=True)
    assert model.backbone[1].weight.requires_grad
    assert not model.backbone[3].weight.requires_grad


def test_unfreeze_and_add_param_group_function(tmpdir):
    """Test unfreeze_and_add_param_group properly unfreeze parameters and add to the correct param_group"""

    seed_everything(42)

    class FreezeModel(LightningModule):

        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=False),
                nn.BatchNorm1d(32),
            )

    model = FreezeModel()
    optimizer = SGD(model.backbone[0].parameters(), lr=0.01)

    with pytest.warns(UserWarning, match="The provided params to be freezed already"):
        BaseFinetuning.unfreeze_and_add_param_group(model.backbone[0], optimizer=optimizer)
    assert optimizer.param_groups[0]["lr"] == 0.01

    model.backbone[1].weight.requires_grad = False
    BaseFinetuning.unfreeze_and_add_param_group(model.backbone[1], optimizer=optimizer)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] == 0.001
    assert torch.equal(optimizer.param_groups[1]["params"][0], model.backbone[1].weight)
    assert model.backbone[1].weight.requires_grad

    with pytest.warns(UserWarning, match="The provided params to be freezed already"):
        BaseFinetuning.unfreeze_and_add_param_group(model, optimizer=optimizer, lr=100, train_bn=False)
    assert len(optimizer.param_groups) == 3
    assert optimizer.param_groups[2]["lr"] == 100
    assert len(optimizer.param_groups[2]["params"]) == 3
    for group_idx, group in enumerate(optimizer.param_groups):
        if group_idx == 0:
            assert torch.equal(optimizer.param_groups[0]["params"][0], model.backbone[0].weight)
        if group_idx == 2:
            assert torch.equal(optimizer.param_groups[2]["params"][0], model.backbone[2].weight)
            assert torch.equal(optimizer.param_groups[2]["params"][1], model.backbone[3].weight)
            assert torch.equal(optimizer.param_groups[2]["params"][2], model.backbone[4].weight)


def test_on_before_accelerator_backend_setup(tmpdir):
    """
    `on_before_accelerator_backend_setup` hook is used by finetuning callbacks to freeze the model before
    before configure_optimizers function call.
    """

    class TestCallback(Callback):

        def on_before_accelerator_backend_setup(self, trainer, pl_module):
            pl_module.on_before_accelerator_backend_setup_called = True

    class TestModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.on_before_accelerator_backend_setup_called = False

        def configure_optimizers(self):
            assert self.on_before_accelerator_backend_setup_called
            return super().configure_optimizers()

    model = TestModel()
    callback = TestCallback()

    trainer = Trainer(default_root_dir=tmpdir, callbacks=[callback], fast_dev_run=True)
    trainer.fit(model)
