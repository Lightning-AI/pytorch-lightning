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
Tests the evaluation loop
"""

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from tests.helpers.deterministic_model import DeterministicModel


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

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.train_loop.run_training_batch(batch, batch_idx, 0)
    assert out.signal == 0
    assert len(out.grad_norm_dict) == 0 and isinstance(out.grad_norm_dict, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out['minimize'], torch.Tensor)
    assert train_step_out['minimize'].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch,
        batch_idx,
        0,
        trainer.optimizers[0],
        hiddens=None,
    )
    assert opt_closure_result['loss'].item() == 171


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

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.train_loop.run_training_batch(batch, batch_idx, 0)
    assert out.signal == 0
    assert len(out.grad_norm_dict) == 0 and isinstance(out.grad_norm_dict, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out['minimize'], torch.Tensor)
    assert train_step_out['minimize'].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch, batch_idx, 0, trainer.optimizers[0], hiddens=None
    )
    assert opt_closure_result['loss'].item() == 171


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

    trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert model.validation_epoch_end_called


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

    trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert model.validation_step_end_called
    assert model.validation_epoch_end_called
