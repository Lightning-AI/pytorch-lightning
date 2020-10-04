"""
Tests to ensure that the training loop works with a dict (1.0)
"""
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel
import os
import torch


def test__training_step__flow_scalar(tmpdir):
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

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called
    assert not model.training_step_end_called
    assert not model.training_epoch_end_called


def test__training_step__tr_step_end__flow_scalar(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

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

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called
    assert model.training_step_end_called
    assert not model.training_epoch_end_called


def test__training_step__epoch_end__flow_scalar(tmpdir):
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

        def training_epoch_end(self, outputs):
            self.training_epoch_end_called = True

            # verify we saw the current num of batches
            assert len(outputs) == 2

            for b in outputs:
                # time = 1
                assert len(b) == 1
                assert 'loss' in b
                assert isinstance(b, dict)

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called
    assert not model.training_step_end_called
    assert model.training_epoch_end_called


def test__training_step__step_end__epoch_end__flow_scalar(tmpdir):
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

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.val_dataloader = None

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
    assert model.training_step_called
    assert model.training_step_end_called
    assert model.training_epoch_end_called
