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
"""
Tests to ensure that the training loop works with a dict (1.0)
"""
import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from tests.base.deterministic_model import DeterministicModel


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__eval_step__flow(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            self.validation_step_called = True
            if batch_idx == 0:
                out = ['1', 2, torch.tensor(2)]
            if batch_idx > 0:
                out = {'something': 'random'}
            return out

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__eval_step__eval_step_end__flow(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            self.validation_step_called = True
            if batch_idx == 0:
                out = ['1', 2, torch.tensor(2)]
            if batch_idx > 0:
                out = {'something': 'random'}
            self.last_out = out
            return out

        def validation_step_end(self, out):
            self.validation_step_end_called = True
            assert self.last_out == out
            return out

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert model.validation_step_end_called
    assert not model.validation_epoch_end_called


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__eval_step__epoch_end__flow(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            self.validation_step_called = True
            if batch_idx == 0:
                out = ['1', 2, torch.tensor(2)]
                self.out_a = out
            if batch_idx > 0:
                out = {'something': 'random'}
                self.out_b = out
            return out

        def validation_epoch_end(self, outputs):
            self.validation_epoch_end_called = True
            assert len(outputs) == 2

            out_a = outputs[0]
            out_b = outputs[1]

            assert out_a == self.out_a
            assert out_b == self.out_b

            return {'no returns needed'}

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.validation_step_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )

    with pytest.warns(UserWarning, match=r".*should not return anything as of 9.1.*"):
        trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert model.validation_epoch_end_called


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__validation_step__step_end__epoch_end__flow(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            self.validation_step_called = True
            if batch_idx == 0:
                out = ['1', 2, torch.tensor(2)]
                self.out_a = out
            if batch_idx > 0:
                out = {'something': 'random'}
                self.out_b = out
            self.last_out = out
            return out

        def validation_step_end(self, out):
            self.validation_step_end_called = True
            assert self.last_out == out
            return out

        def validation_epoch_end(self, outputs):
            self.validation_epoch_end_called = True
            assert len(outputs) == 2

            out_a = outputs[0]
            out_b = outputs[1]

            assert out_a == self.out_a
            assert out_b == self.out_b

            return {'no returns needed'}

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )

    with pytest.warns(UserWarning, match=r".*should not return anything as of 9.1.*"):
        trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert model.validation_step_end_called
    assert model.validation_epoch_end_called
