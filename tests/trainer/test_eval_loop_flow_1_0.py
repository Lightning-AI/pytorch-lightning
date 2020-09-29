"""
Tests to ensure that the training loop works with a dict (1.0)
"""
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel
import os
import torch
import pytest


def test__eval_step__flow(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

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

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        row_log_interval=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called


def test__eval_step__eval_step_end__flow(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

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

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        row_log_interval=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert model.validation_step_end_called
    assert not model.validation_epoch_end_called


def test__eval_step__epoch_end__flow(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

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

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.validation_step_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        row_log_interval=1,
        weights_summary=None,
    )

    with pytest.warns(UserWarning, match=r".*should not return anything as of 9.1.*"):
        trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert model.validation_epoch_end_called


def test__validation_step__step_end__epoch_end__flow(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

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

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        row_log_interval=1,
        weights_summary=None,
    )

    with pytest.warns(UserWarning, match=r".*should not return anything as of 9.1.*"):
        trainer.fit(model)

    # make sure correct steps were called
    assert model.validation_step_called
    assert model.validation_step_end_called
    assert model.validation_epoch_end_called
