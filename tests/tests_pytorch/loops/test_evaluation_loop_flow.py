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
"""Tests the evaluation loop."""

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.trainer.states import RunningStage
from torch import Tensor

from tests_pytorch.helpers.deterministic_model import DeterministicModel


def test__eval_step__flow(tmpdir):
    """Tests that only training_step can be used."""

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            self.validation_step_called = True
            if batch_idx == 0:
                out = ["1", 2, torch.tensor(2)]
            if batch_idx > 0:
                out = {"something": "random"}
            return out

        def backward(self, loss):
            return LightningModule.backward(self, loss)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called

    # simulate training manually
    trainer.state.stage = RunningStage.TRAINING
    kwargs = {"batch": next(iter(model.train_dataloader())), "batch_idx": 0}
    train_step_out = trainer.fit_loop.epoch_loop.automatic_optimization.run(trainer.optimizers[0], 0, kwargs)

    assert isinstance(train_step_out["loss"], Tensor)
    assert train_step_out["loss"].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure = trainer.fit_loop.epoch_loop.automatic_optimization._make_closure(kwargs, trainer.optimizers[0], 0)
    opt_closure_result = opt_closure()
    assert opt_closure_result.item() == 171


def test__eval_step__epoch_end__flow(tmpdir):
    """Tests that only training_step can be used."""

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            self.validation_step_called = True
            if batch_idx == 0:
                out = ["1", 2, torch.tensor(2)]
                self.out_a = out
            if batch_idx > 0:
                out = {"something": "random"}
                self.out_b = out
            return out

        def backward(self, loss):
            return LightningModule.backward(self, loss)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        enable_model_summary=False,
    )

    trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
