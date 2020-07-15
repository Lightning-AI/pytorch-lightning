"""
Tests to ensure that the training loop works with a dict
"""
import os
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel
from pytorch_lightning.core.step_result import Result, TrainResult, EvalResult


def test_training_step_result(tmpdir):
    """
    Tests that only training_step can be used with TrainResult
    Makes sure that things are routed to pbar, loggers and loss accordingly
    """
    # enable internal debugging actions
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_return
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert not model.training_epoch_end_called

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert out.batch_log_metrics['log_and_pbar_acc1'] == 12.0
    assert out.batch_log_metrics['log_acc2'] == 7.0

    train_step_out = out.training_step_output_for_epoch_end
    assert isinstance(train_step_out, TrainResult)

    assert 'minimize' in train_step_out
    assert 'log_and_pbar_acc1' in train_step_out
    assert 'log_acc2' in train_step_out

    # make sure we are using the correct metrics for callbacks
    assert trainer.callback_metrics['early_stop_on'] == 171
    assert trainer.callback_metrics['checkpoint_on'] == 171

    # make sure pbar metrics are correct
    assert trainer.progress_bar_metrics['log_and_pbar_acc1'] == 12
    assert trainer.progress_bar_metrics['pbar_acc3'] == 17
    assert 'log_acc2' not in trainer.progress_bar_metrics

    # make sure correct metrics are logged
    assert len(trainer.debug_logged_metrics) == 1
    logged_metrics = trainer.debug_logged_metrics[0]
    assert logged_metrics['log_and_pbar_acc1'] == 12.0
    assert logged_metrics['log_acc2'] == 7.0
    assert 'pbar_acc3' not in logged_metrics
    assert len(logged_metrics) == 3

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.optimizer_closure(batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'] == (42.0 * 3) + (15.0 * 3)


def test_training_step_epoch_end_result(tmpdir):
    """
    Makes sure training_step and epoch_end can be used with Results (without batch_end)
    """
    # TODO: implement
    pass
