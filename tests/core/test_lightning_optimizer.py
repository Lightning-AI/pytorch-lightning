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
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities import BOLT_AVAILABLE
from tests.base.boring_model import BoringModel, RandomDataset, RandomDictDataset, RandomDictStringDataset

if BOLT_AVAILABLE:
    from pl_bolts.optimizers.lars_scheduling import LARSWrapper
    from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


def test_lightning_optimizer(tmpdir):
    """
    Test that optimizer are correctly wrapped by our LightningOptimizer
    """
    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            # optimizer = LightningOptimizer(self.trainer, optimizer)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        enable_pl_optimizer=True,
    )
    trainer.fit(model)

    groups = "{'dampening': 0, 'initial_lr': 0.1, 'lr': 0.01, 'momentum': 0, 'nesterov': False, 'weight_decay': 0}"
    expected = f"LightningSGD(groups=[{groups}])"
    assert trainer.optimizers[0].__repr__() == expected


def test_lightning_optimizer_from_user(tmpdir):
    """
    Test that the user can use our LightningOptimizer. Not recommended.
    """

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            optimizer = LightningOptimizer(optimizer)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        enable_pl_optimizer=True,
    )
    trainer.fit(model)

    groups = "{'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'initial_lr': 0.1, 'lr': 0.01, 'weight_decay': 0}"
    expected = f"LightningAdam(groups=[{groups}])"
    assert trainer.optimizers[0].__repr__() == expected


@patch("torch.optim.Adam.step")
@patch("torch.optim.SGD.step")
def test_lightning_optimizer_manual_optimization(mock_sgd_step, mock_adam_step, tmpdir):
    """
    Test that the user can use our LightningOptimizer. Not recommended for now.
    """
    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            (opt_1, opt_2) = self.optimizers()
            assert isinstance(opt_1, LightningOptimizer)
            assert isinstance(opt_2, LightningOptimizer)

            output = self.layer(batch)
            loss_1 = self.loss(batch, output)
            self.manual_backward(loss_1, opt_1)
            opt_1.step(idx="1")

            def closure():
                output = self.layer(batch)
                loss_2 = self.loss(batch, output)
                self.manual_backward(loss_2, opt_2)
            opt_2.step(closure=closure, idx="2")

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            optimizer_1 = LightningOptimizer(optimizer_1, 4)

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1, optimizer_2], [lr_scheduler]

    model = TestModel()
    model.training_step_end = None
    model.training_epoch_end = None
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=8,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        automatic_optimization=False,
        enable_pl_optimizer=True)
    trainer.fit(model)

    assert len(mock_sgd_step.mock_calls) == 2
    assert len(mock_adam_step.mock_calls) == 8


@patch("torch.optim.Adam.step")
@patch("torch.optim.SGD.step")
def test_lightning_optimizer_manual_optimization_and_accumulated_gradients(mock_sgd_step, mock_adam_step, tmpdir):
    """
    Test that the user can use our LightningOptimizer. Not recommended.
    """
    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            (opt_1, opt_2) = self.optimizers()
            assert isinstance(opt_1, LightningOptimizer)
            assert isinstance(opt_2, LightningOptimizer)

            output = self.layer(batch)
            loss_1 = self.loss(batch, output)
            self.manual_backward(loss_1, opt_1)
            opt_1.step(idx="1")

            def closure():
                output = self.layer(batch)
                loss_2 = self.loss(batch, output)
                self.manual_backward(loss_2, opt_2)
            opt_2.step(closure=closure, idx="2")

        def configure_optimizers(self):
            optimizer_1 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            optimizer_1 = LightningOptimizer(optimizer_1, 4)

            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            return [optimizer_1, optimizer_2], [lr_scheduler]

    model = TestModel()
    model.training_step_end = None
    model.training_epoch_end = None
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=8,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
        automatic_optimization=False,
        accumulate_grad_batches=2,
        enable_pl_optimizer=True,
    )
    trainer.fit(model)

    assert len(mock_sgd_step.mock_calls) == 2
    assert len(mock_adam_step.mock_calls) == 4


def test_state(tmpdir):
    model = torch.nn.Linear(3, 4)
    optimizer = torch.optim.Adam(model.parameters())
    lightning_optimizer = LightningOptimizer(optimizer)
    assert isinstance(lightning_optimizer, LightningOptimizer)
    assert isinstance(lightning_optimizer, Adam)
    assert isinstance(lightning_optimizer, Optimizer)
    lightning_dict = {}
    special_attrs = ["_accumulate_grad_batches", "_optimizer", "_optimizer_idx",
                     "_trainer", "_use_accumulate_grad_batches_from_trainer", "_lightning_step"]
    for k, v in lightning_optimizer.__dict__.items():
        if k not in special_attrs:
            lightning_dict[k] = v
    assert lightning_dict == optimizer.__dict__
    assert optimizer.state_dict() == lightning_optimizer.state_dict()
    assert optimizer.state == lightning_optimizer.state


def test_lightning_optimizer_state(tmpdir):
    class CheckpointEveryNSteps(pl.Callback):
        """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
        """

        def __init__(
            self,
            save_step_frequency,
            prefix="latest-Checkpoint",
            use_modelcheckpoint_filename=False,
        ):
            """
            Args:
                save_step_frequency: how often to save in steps
                prefix: add a prefix to the name, only used if
                    use_modelcheckpoint_filename=False
                use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                    default filename, don't use ours.
            """
            self.save_step_frequency = save_step_frequency
            self.prefix = prefix
            self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

        def on_batch_end(self, trainer: pl.Trainer, _):
            """ Check if we should save a checkpoint after every train batch """
            global_step = trainer.global_step
            if global_step % self.save_step_frequency == 0:
                if self.use_modelcheckpoint_filename:
                    filename = trainer.checkpoint_callback.filename
                else:
                    filename = "{}.ckpt".format(self.prefix)
                ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
                trainer.save_checkpoint(ckpt_path)

    class TestModel(BoringModel):

        def on_train_epoch_start(self) -> None:
            print('override any method to prove your bug')

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

            optimizer = LARSWrapper(optimizer)
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=1,
                max_epochs=20
            )
            return [optimizer], [scheduler]

    train_data = torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=1)
    val_data = torch.utils.data.DataLoader(RandomDataset(32, 64),batch_size=1)
    test_data = torch.utils.data.DataLoader(RandomDataset(32, 64),batch_size=1)

    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        mode='min',
        filepath=tmpdir
    )

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=10,
        weights_summary=None,
        accelerator='ddp',
        log_every_n_steps=1,
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        callbacks=[CheckpointEveryNSteps(1)]
    )
    trainer.fit(model, train_data, val_data)
