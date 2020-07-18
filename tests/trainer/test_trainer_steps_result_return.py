"""
Tests to ensure that the training loop works with a dict
"""
import os
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel
from pytorch_lightning.core.step_result import Result, TrainResult, EvalResult


# TODOs:
# make checkpoint and early stopping use the correct metrics
# make sure step_ends receive a plain dict
# same for epoch_end
# make sure to auto-reduce when no epoch_end is implemented

def test_training_step_result_log_step_only(tmpdir):
    """
    Tests that only training_step can be used with TrainResult
    Makes sure that things are routed to pbar, loggers and loss accordingly

    Makes sure pbar and logs happen on step only when requested
    """
    # enable internal debugging actions
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_log_step_only
    model.training_step_end = None
    model.training_epoch_end = None
    model.val_dataloader = None

    batches = 3
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        row_log_interval=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert not model.training_epoch_end_called

    # make sure correct metrics are logged (one per batch step as requested)
    assert len(trainer.debug_logged_metrics) == batches + 1
    for batch_idx, logged_metrics in enumerate(trainer.debug_logged_metrics[:-1]):
        assert logged_metrics[f'step_log_and_pbar_acc1_b{batch_idx}'] == 11.0
        assert logged_metrics[f'step_log_acc2_b{batch_idx}'] == 12.0
        assert f'step_pbar_acc3_b{batch_idx}' not in logged_metrics
        assert len(logged_metrics) == 3

    # make sure we are using the correct metrics for callbacks
    assert trainer.callback_metrics['early_stop_on'] == 171
    assert trainer.callback_metrics['checkpoint_on'] == 171

    # make sure pbar metrics are correct ang log metrics did not leak
    for batch_idx in range(batches):
        assert trainer.progress_bar_metrics[f'step_log_and_pbar_acc1_b{batch_idx}'] == 11
        assert trainer.progress_bar_metrics[f'step_pbar_acc3_b{batch_idx}'] == 13
        assert f'step_log_acc2_b{batch_idx}' not in trainer.progress_bar_metrics

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert out.batch_log_metrics[f'step_log_and_pbar_acc1_b{batch_idx}'] == 11.0
    assert out.batch_log_metrics[f'step_log_acc2_b{batch_idx}'] == 12.0

    train_step_out = out.training_step_output_for_epoch_end
    assert isinstance(train_step_out, TrainResult)

    assert 'minimize' in train_step_out
    assert f'step_log_and_pbar_acc1_b{batch_idx}' in train_step_out
    assert f'step_log_acc2_b{batch_idx}' in train_step_out

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.optimizer_closure(batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'] == (42.0 * 3) + (15.0 * 3)

test_training_step_result_log_step_only('')

def test_training_step_auto_reduce(tmpdir):
    # TODO: test that it gets reduced on epoch end
    # TODO: test that on batch end gets reduced

    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_return
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)



def test_training_step_epoch_end_result(tmpdir):
    """
    Makes sure training_step and epoch_end can be used with Results (without batch_end)
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_return
    model.training_epoch_end = model.training_epoch_end_return
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert model.training_epoch_end_called

    # make sure correct metrics were logged
    logged_metrics = trainer.debug_logged_metrics[-1]
    assert logged_metrics['log_and_pbar_acc1'] == 23.0
    assert logged_metrics['log_acc2'] == 18.0
    assert logged_metrics['epoch_end_log_acc'] == 1212.0
    assert logged_metrics['epoch_end_log_pbar_acc'] == 1214.0
    assert 'epoch_end_pbar_acc' not in logged_metrics

    # make sure pbar metrics are correct
    assert trainer.progress_bar_metrics['log_and_pbar_acc1'] == 23.0
    assert trainer.progress_bar_metrics['pbar_acc3'] == 28.0
    assert trainer.progress_bar_metrics['epoch_end_pbar_acc'] == 1213.0
    assert trainer.progress_bar_metrics['epoch_end_log_pbar_acc'] == 1214.0
    assert 'epoch_end_log_acc' not in trainer.progress_bar_metrics
    assert 'log_acc2' not in trainer.progress_bar_metrics

    # make sure callback metrics didn't change
    assert trainer.callback_metrics['early_stop_on'] == 171
    assert trainer.callback_metrics['checkpoint_on'] == 171
