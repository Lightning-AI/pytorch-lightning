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
"""Tests to ensure that the training loop works with a dict (1.0)"""

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.core.module import LightningModule

from tests_pytorch.helpers.deterministic_model import DeterministicModel


def test__training_step__flow_dict(tmpdir):
    """Tests that only training_step can be used."""

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return {"loss": acc, "random_things": [1, "a", torch.tensor(2)]}

        def backward(self, loss):
            return LightningModule.backward(self, loss)

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called


def test__training_step__tr_batch_end__flow_dict(tmpdir):
    """Tests that only training_step can be used."""

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            self.out = {"loss": acc, "random_things": [1, "a", torch.tensor(2)]}
            return self.out

        def on_train_batch_end(self, tr_step_output, *_):
            assert self.count_num_graphs(tr_step_output) == 0

        def backward(self, loss):
            return LightningModule.backward(self, loss)

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called


def test__training_step__epoch_end__flow_dict(tmpdir):
    """Tests that only training_step can be used."""

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx

            self.training_step_called = True
            return {"loss": acc, "random_things": [1, "a", torch.tensor(2)], "batch_idx": batch_idx}

        def backward(self, loss):
            return LightningModule.backward(self, loss)

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called


def test__training_step__batch_end__epoch_end__flow_dict(tmpdir):
    """Tests that only training_step can be used."""

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx

            self.training_step_called = True
            self.out = {"loss": acc, "random_things": [1, "a", torch.tensor(2)], "batch_idx": batch_idx}
            return self.out

        def on_train_batch_end(self, tr_step_output, *_):
            assert self.count_num_graphs(tr_step_output) == 0

        def backward(self, loss):
            return LightningModule.backward(self, loss)

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called
