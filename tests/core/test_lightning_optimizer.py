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

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from tests.base.boring_model import BoringModel, RandomDictDataset, RandomDictStringDataset


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
