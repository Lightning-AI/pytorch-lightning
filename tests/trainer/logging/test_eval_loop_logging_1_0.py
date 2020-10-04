"""
Tests to ensure that the training loop works with a dict (1.0)
"""
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from tests.base.deterministic_model import DeterministicModel
from tests.base import SimpleModule, BoringModel
import os
import torch
import pytest


def test__validation_step__log(tmpdir):
    """
    Tests that validation_step can log
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('a', acc, on_step=True, on_epoch=True)
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('b', acc, on_step=True, on_epoch=True)
            self.training_step_called = True

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

    # make sure all the metrics are available for callbacks
    expected_logged_metrics = {
        'a',
        'step_a',
        'epoch_a',
        'b',
        'step_b/epoch_0',
        'step_b/epoch_1',
        'epoch_b',
        'epoch',
    }
    logged_metrics = set(trainer.logged_metrics.keys())
    assert expected_logged_metrics == logged_metrics

    # we don't want to enable val metrics during steps because it is not something that users should do
    # on purpose DO NOT allow step_b... it's silly to monitor val step metrics
    expected_cb_metrics = {'a', 'b', 'epoch_a', 'epoch_b', 'step_a'}
    callback_metrics = set(trainer.callback_metrics.keys())
    assert expected_cb_metrics == callback_metrics


def test__validation_step__step_end__epoch_end__log(tmpdir):
    """
    Tests that validation_step can log
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('a', acc)
            self.log('b', acc, on_step=True, on_epoch=True)
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('c', acc)
            self.log('d', acc, on_step=True, on_epoch=True)
            self.validation_step_called = True
            return acc

        def validation_step_end(self, acc):
            self.validation_step_end_called = True
            self.log('e', acc)
            self.log('f', acc, on_step=True, on_epoch=True)
            return ['random_thing']

        def validation_epoch_end(self, outputs):
            self.log('g', torch.tensor(2, device=self.device), on_epoch=True)
            self.validation_epoch_end_called = True

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
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics.keys())
    expected_logged_metrics = {
        'epoch',
        'a',
        'b',
        'step_b',
        'epoch_b',
        'c',
        'd',
        'step_d/epoch_0',
        'step_d/epoch_1',
        'epoch_d',
        'e',
        'f',
        'step_f/epoch_0',
        'step_f/epoch_1',
        'epoch_f',
        'g',
    }
    assert expected_logged_metrics == logged_metrics

    progress_bar_metrics = set(trainer.progress_bar_metrics.keys())
    expected_pbar_metrics = set()
    assert expected_pbar_metrics == progress_bar_metrics

    # we don't want to enable val metrics during steps because it is not something that users should do
    callback_metrics = set(trainer.callback_metrics.keys())
    callback_metrics.remove('debug_epoch')
    expected_cb_metrics = {'a', 'b', 'c', 'd', 'e', 'epoch_b', 'epoch_d', 'epoch_f', 'f', 'g', 'step_b'}
    assert expected_cb_metrics == callback_metrics


@pytest.mark.parametrize(['batches', 'log_interval', 'max_epochs'], [(1, 1, 1), (64, 32, 2)])
def test_eval_epoch_logging(tmpdir, batches, log_interval, max_epochs):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):
        def validation_epoch_end(self, outputs):
            self.log('c', torch.tensor(2), on_epoch=True, prog_bar=True, logger=True)
            self.log('d/e/f', 2)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        max_epochs=max_epochs,
        row_log_interval=log_interval,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics.keys())
    expected_logged_metrics = {
        'c',
        'd/e/f',
    }
    assert logged_metrics == expected_logged_metrics

    pbar_metrics = set(trainer.progress_bar_metrics.keys())
    expected_pbar_metrics = {'c'}
    assert pbar_metrics == expected_pbar_metrics

    callback_metrics = set(trainer.callback_metrics.keys())
    expected_callback_metrics = {'early_stop_on', 'checkpoint_on'}
    expected_callback_metrics = expected_callback_metrics.union(logged_metrics)
    expected_callback_metrics = expected_callback_metrics.union(pbar_metrics)
    assert callback_metrics == expected_callback_metrics

    # assert the loggers received the expected number
    assert len(trainer.dev_debugger.logged_metrics) == max_epochs


def test_monitor_val_epoch_end(tmpdir):
    epoch_min_loss_override = 0
    model = SimpleModule()
    checkpoint_callback = callbacks.ModelCheckpoint(save_top_k=1, monitor="avg_val_loss")
    trainer = Trainer(
        max_epochs=epoch_min_loss_override + 2,
        logger=False,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)
