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
from collections import OrderedDict

import pytest
import torch
from torch import nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import BackboneFinetuning, BaseFinetuning, ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from tests.helpers import BoringModel, RandomDataset


class TestBackboneFinetuningCallback(BackboneFinetuning):

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if self.unfreeze_backbone_at_epoch <= epoch:
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']
            backbone_lr = self.previous_backbone_lr
            if epoch < 6:
                assert backbone_lr <= current_lr
            else:
                assert backbone_lr == current_lr


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

    model = FinetuningBoringModel()
    callback = TestBackboneFinetuningCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    trainer = Trainer(
        limit_train_batches=4,
        default_root_dir=tmpdir,
        callbacks=[callback],
        max_epochs=8,
    )
    trainer.fit(model)

    assert model.backbone.has_been_used


class TestBackboneFinetuningWarningCallback(BackboneFinetuning):

    def finetune_function(self, pl_module, epoch: int, optimizer, opt_idx: int):
        """Called when the epoch begins."""

        if epoch == 0:
            self.unfreeze_and_add_param_group(
                pl_module.backbone, optimizer, 0.1, train_bn=self.train_bn, initial_denom_lr=self.initial_denom_lr
            )


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

    chk = ModelCheckpoint(dirpath=tmpdir, save_last=True)

    model = FinetuningBoringModel()
    model.validation_step = None
    callback = TestBackboneFinetuningWarningCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    with pytest.warns(UserWarning, match="Did you init your optimizer in"):
        trainer = Trainer(
            limit_train_batches=1,
            default_root_dir=tmpdir,
            callbacks=[callback, chk],
            max_epochs=2,
        )
        trainer.fit(model)

    assert model.backbone.has_been_used
    trainer = Trainer(max_epochs=3, resume_from_checkpoint=chk.last_model_path)
    trainer.fit(model)


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


class OnEpochLayerFinetuning(BaseFinetuning):

    def freeze_before_training(self, pl_module: LightningModule):
        self.freeze(pl_module.layer)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        self.unfreeze_and_add_param_group(pl_module.layer[epoch + 1], optimizer)


def test_base_finetuning_internal_state(tmpdir):
    """Test the param_groups updates are properly saved within the internal state of the BaseFinetuning Callbacks"""

    seed_everything(42)

    class FreezeModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.layer = nn.Sequential(
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=True),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 32, bias=True),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, 2, bias=True),
            )

        def forward(self, x):
            return self.layer(x)

        def configure_optimizers(self):
            return torch.optim.SGD(self.layer[0].parameters(), lr=0.1)

    cb = OnEpochLayerFinetuning()
    chk = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    model = FreezeModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=5, limit_train_batches=1, callbacks=[cb, chk])
    trainer.fit(model)
    assert len(cb._internal_state[0]) == 6
    assert cb._internal_state[0][0]["params"] == ['layer.0.weight']
    assert cb._internal_state[0][1]["params"] == ['layer.1.weight', 'layer.1.bias']
    assert cb._internal_state[0][2]["params"] == ['layer.2.weight']
    assert cb._internal_state[0][3]["params"] == ['layer.3.weight', 'layer.3.bias']
    assert cb._internal_state[0][4]["params"] == ['layer.4.weight']
    assert cb._internal_state[0][5]["params"] == ['layer.5.weight', 'layer.5.bias']

    model = FreezeModel()
    cb = OnEpochLayerFinetuning()
    trainer = Trainer(max_epochs=10, resume_from_checkpoint=chk.last_model_path, callbacks=[cb])
    with pytest.raises(IndexError, match="index 6 is out of range"):
        trainer.fit(model)


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


def test_deep_nested_model():

    class ConvBlock(nn.Module):

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3)
            self.act = nn.ReLU()
            self.bn = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.act(x)
            return self.bn(x)

    model = nn.Sequential(
        OrderedDict([
            ("encoder", nn.Sequential(ConvBlock(3, 64), ConvBlock(64, 128))),
            ("decoder", ConvBlock(128, 10)),
        ])
    )

    # There's 9 leaf layers in that model
    assert len(BaseFinetuning.flatten_modules(model)) == 9

    BaseFinetuning.freeze(model.encoder, train_bn=True)
    assert not model.encoder[0].conv.weight.requires_grad
    assert model.encoder[0].bn.weight.requires_grad

    BaseFinetuning.make_trainable(model)
    encoder_params = list(BaseFinetuning.filter_params(model.encoder, train_bn=True))
    # The 8 parameters of the encoder are:
    # conv0.weight, conv0.bias, bn0.weight, bn0.bias
    # conv1.weight, conv1.bias, bn1.weight, bn1.bias
    assert len(encoder_params) == 8
