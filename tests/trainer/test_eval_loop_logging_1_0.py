"""
Tests to ensure that the training loop works with a dict (1.0)
"""
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel
import os
import torch


def test__validation_step__log(tmpdir):
    """
    Tests that validation_step can log
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('train_step_acc', acc, on_step=True, on_epoch=True)
            self.training_step_called = True
            return acc

        def validation_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('val_step_acc', acc, on_step=True, on_epoch=True)
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
        'epoch',
        'train_step_acc', 'step_train_step_acc', 'epoch_train_step_acc',
        'val_step_acc/epoch_0', 'val_step_acc/epoch_1',
        'step_val_step_acc/epoch_0', 'step_val_step_acc/epoch_1',
    }
    logged_metrics = set(trainer.logged_metrics.keys())
    assert expected_logged_metrics == logged_metrics

    # we don't want to enable val metrics during steps because it is not something that users should do
    expected_cb_metrics = [
        'train_step_acc', 'step_train_step_acc', 'epoch_train_step_acc',
    ]
    expected_cb_metrics = set(expected_cb_metrics)
    callback_metrics = set(trainer.callback_metrics.keys())
    assert expected_cb_metrics == callback_metrics
