"""
Tests to ensure that the training loop works with a dict (1.0)
"""
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel
import os
import torch


def test__training_step__log(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('step_acc', acc, on_step=True, on_epoch=False)
            self.log('epoch_acc', acc, on_step=False, on_epoch=True)
            self.log('no_prefix_step_epoch_acc', acc, on_step=True, on_epoch=True)
            self.log('pbar_step_acc', acc, on_step=True, prog_bar=True, on_epoch=False, logger=False)
            self.log('pbar_epoch_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            self.log('pbar_step_epoch_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=False)

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

    # make sure all the metrics are available for callbacks
    metrics = [
        'step_acc',
        'epoch_acc',
        'no_prefix_step_epoch_acc', 'step_no_prefix_step_epoch_acc', 'epoch_no_prefix_step_epoch_acc',
        'pbar_step_acc',
        'pbar_epoch_acc',
        'pbar_step_epoch_acc', 'step_pbar_step_epoch_acc', 'epoch_pbar_step_epoch_acc',
    ]
    expected_metrics = set(metrics + ['debug_epoch'])
    callback_metrics = set(trainer.callback_metrics.keys())
    assert expected_metrics == callback_metrics


def test__training_step__epoch_end__log(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('step_acc', acc, on_step=True, on_epoch=False)
            self.log('epoch_acc', acc, on_step=False, on_epoch=True)
            self.log('no_prefix_step_epoch_acc', acc, on_step=True, on_epoch=True)
            self.log('pbar_step_acc', acc, on_step=True, prog_bar=True, on_epoch=False, logger=False)
            self.log('pbar_epoch_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)
            self.log('pbar_step_epoch_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=False)

            self.training_step_called = True
            return acc

        def training_epoch_end(self, outputs):
            self.training_epoch_end_called = True
            # logging is independent of epoch_end loops
            self.log('custom_epoch_end_metric', torch.tensor(37.2))

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

    # make sure all the metrics are available for callbacks
    metrics = [
        'step_acc',
        'epoch_acc',
        'no_prefix_step_epoch_acc', 'step_no_prefix_step_epoch_acc', 'epoch_no_prefix_step_epoch_acc',
        'pbar_step_acc',
        'pbar_epoch_acc',
        'pbar_step_epoch_acc', 'step_pbar_step_epoch_acc', 'epoch_pbar_step_epoch_acc',
        'custom_epoch_end_metric'
    ]
    expected_metrics = set(metrics + ['debug_epoch'])
    callback_metrics = set(trainer.callback_metrics.keys())
    assert expected_metrics == callback_metrics

    # verify global steps were correctly called

    # epoch 0
    assert trainer.dev_debugger.logged_metrics[0]['global_step'] == 0
    assert trainer.dev_debugger.logged_metrics[1]['global_step'] == 1
    assert trainer.dev_debugger.logged_metrics[2]['global_step'] == 1

    # epoch 1
    assert trainer.dev_debugger.logged_metrics[3]['global_step'] == 2
    assert trainer.dev_debugger.logged_metrics[4]['global_step'] == 3
    assert trainer.dev_debugger.logged_metrics[5]['global_step'] == 3
