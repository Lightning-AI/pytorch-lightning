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
from tests.base.boring_model import BoringModel
from tests.base.deterministic_model import DeterministicModel


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__training_step__flow_scalar(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            return acc

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called
    assert not model.training_step_end_called
    assert not model.training_epoch_end_called


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__training_step__tr_step_end__flow_scalar(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.training_step_called = True
            self.out = acc
            return acc

        def training_step_end(self, tr_step_output):
            assert self.out == tr_step_output
            assert self.count_num_graphs({'loss': tr_step_output}) == 1
            self.training_step_end_called = True
            return tr_step_output

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called
    assert model.training_step_end_called
    assert not model.training_epoch_end_called


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__training_step__epoch_end__flow_scalar(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx

            self.training_step_called = True
            return acc

        def training_epoch_end(self, outputs):
            self.training_epoch_end_called = True

            # verify we saw the current num of batches
            assert len(outputs) == 2

            for b in outputs:
                # time = 1
                assert len(b) == 1
                assert 'loss' in b
                assert isinstance(b, dict)

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called
    assert not model.training_step_end_called
    assert model.training_epoch_end_called


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test__training_step__step_end__epoch_end__flow_scalar(tmpdir):
    """
    Tests that only training_step can be used
    """

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx

            self.training_step_called = True
            return acc

        def training_step_end(self, tr_step_output):
            assert isinstance(tr_step_output, torch.Tensor)
            assert self.count_num_graphs({'loss': tr_step_output}) == 1
            self.training_step_end_called = True
            return tr_step_output

        def training_epoch_end(self, outputs):
            self.training_epoch_end_called = True

            # verify we saw the current num of batches
            assert len(outputs) == 2

            for b in outputs:
                # time = 1
                assert len(b) == 1
                assert 'loss' in b
                assert isinstance(b, dict)

        def backward(self, loss, optimizer, optimizer_idx):
            return LightningModule.backward(self, loss, optimizer, optimizer_idx)

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called
    assert model.training_step_end_called
    assert model.training_epoch_end_called


def test_train_step_no_return(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.training_step_called = True
            loss = self.step(batch[0])
            self.log('a', loss, on_step=True, on_epoch=True)

        def training_epoch_end(self, outputs) -> None:
            assert len(outputs) == 0

        def validation_step(self, batch, batch_idx):
            self.validation_step_called = True

        def validation_epoch_end(self, outputs):
            assert len(outputs) == 0

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        log_every_n_steps=1,
        weights_summary=None,
    )

    with pytest.warns(UserWarning, match=r'.*training_step returned None.*'):
        trainer.fit(model)
    assert model.training_step_called
    assert model.validation_step_called


def test_training_step_no_return_when_even(tmpdir):
    """
    Tests correctness when some training steps have been skipped
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            self.training_step_called = True
            loss = self.step(batch[0])
            self.log('a', loss, on_step=True, on_epoch=True)
            return loss if batch_idx % 2 else None

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=4,
        limit_val_batches=1,
        max_epochs=4,
        weights_summary=None,
        logger=False,
        checkpoint_callback=False,
    )

    with pytest.warns(UserWarning, match=r'.*training_step returned None.*'):
        trainer.fit(model)

    # manually check a few batches
    for batch_idx, batch in enumerate(model.train_dataloader()):
        out = trainer.train_loop.run_training_batch(batch, batch_idx, 0)
        if not batch_idx % 2:
            assert out.training_step_output_for_epoch_end == [[]]
        assert out.signal == 0
