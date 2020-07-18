"""
Tests to ensure that the training loop works with a dict
"""
import os
import torch
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel
from pytorch_lightning.core.step_result import Result, TrainResult, EvalResult


# TODOs:
# make checkpoint and early stopping use the correct metrics

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
    assert len(trainer.debug_logged_metrics) == batches
    for batch_idx, logged_metrics in enumerate(trainer.debug_logged_metrics):
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


def test_training_step_result_log_epoch_only(tmpdir):
    """
    Tests that only training_step can be used with TrainResult
    Makes sure that things are routed to pbar, loggers and loss accordingly

    Makes sure pbar and logs happen on epoch only when requested
    """
    # enable internal debugging actions
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_log_epoch_only
    model.training_step_end = None
    model.training_epoch_end = None
    model.val_dataloader = None

    epochs = 3
    batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        row_log_interval=1,
        max_epochs=epochs,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert not model.training_epoch_end_called

    # make sure correct metrics are logged (one per batch step as requested)
    assert len(trainer.debug_logged_metrics) == epochs
    epoch_metrics = trainer.debug_logged_metrics
    assert len(epoch_metrics) == epochs
    for batch_idx, logged_metrics in enumerate(epoch_metrics):
        assert logged_metrics[f'epoch_log_and_pbar_acc1_e{batch_idx}'] == 14.0
        assert logged_metrics[f'epoch_log_acc2_e{batch_idx}'] == 15.0
        assert f'epoch_pbar_acc3_e{batch_idx}' not in logged_metrics
        assert len(logged_metrics) == 3

    # make sure we are using the correct metrics for callbacks
    assert trainer.callback_metrics['early_stop_on'] == 171
    assert trainer.callback_metrics['checkpoint_on'] == 171

    # make sure pbar metrics are correct ang log metrics did not leak
    for epoch_idx in range(epochs):
        assert trainer.progress_bar_metrics[f'epoch_log_and_pbar_acc1_e{epoch_idx}'] == 14
        assert trainer.progress_bar_metrics[f'epoch_pbar_acc3_e{epoch_idx}'] == 16
        assert f'epoch_log_acc2_e{epoch_idx}' not in trainer.progress_bar_metrics

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert len(out.batch_log_metrics) == 0

    train_step_out = out.training_step_output_for_epoch_end
    assert isinstance(train_step_out, TrainResult)

    assert 'minimize' in train_step_out
    assert f'epoch_log_and_pbar_acc1_e{trainer.current_epoch}' in train_step_out
    assert f'epoch_log_acc2_e{trainer.current_epoch}' in train_step_out

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.optimizer_closure(batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'] == (42.0 * 3) + (15.0 * 3)


def test_training_step_result_log_step_and_epoch(tmpdir):
    """
    Tests that only training_step can be used with TrainResult
    Makes sure that things are routed to pbar, loggers and loss accordingly

    Makes sure pbar and logs happen on epoch only when requested
    """
    # enable internal debugging actions
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_log_epoch_and_step
    model.training_step_end = None
    model.training_epoch_end = None
    model.val_dataloader = None

    epochs = 3
    batches = 2
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=batches,
        limit_val_batches=batches,
        row_log_interval=1,
        max_epochs=epochs,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert not model.training_epoch_end_called

    # make sure correct metrics are logged (one per batch step as requested)
    assert len(trainer.debug_logged_metrics) == (epochs * batches) + epochs
    epoch_metrics = trainer.debug_logged_metrics
    epoch_idx = -1
    for i_start in range(0, len(epoch_metrics), batches + 1):
        epoch_idx += 1
        epoch_outputs = epoch_metrics[i_start: i_start + batches + 1]
        mean_vals = {
            'step_epoch_log_and_pbar_acc1': [],
            'step_epoch_log_acc2': []
        }

        # make sure each batch logged the expected value
        for batch_idx in range(len(epoch_outputs) - 1):
            logged_metrics = epoch_outputs[batch_idx]

            expected_val_1 = (5 + batch_idx) * (epoch_idx + 1)
            expected_val_2 = (6 + batch_idx) * (epoch_idx + 1)
            mean_vals['step_epoch_log_and_pbar_acc1'].append(torch.tensor(expected_val_1).float())
            mean_vals['step_epoch_log_acc2'].append(torch.tensor(expected_val_2).float())
            assert logged_metrics['step_epoch_log_and_pbar_acc1'] == expected_val_1
            assert logged_metrics['step_epoch_log_acc2'] == expected_val_2
            assert 'step_epoch_pbar_acc3' not in logged_metrics
            assert len(logged_metrics) == 3

        # make sure the metrics for the epoch end are actual means (the default reduce fx) or all the batches
        epoch_end_metrics = epoch_outputs[-1]
        eval_1 = torch.stack(mean_vals['step_epoch_log_and_pbar_acc1']).mean()
        eval_2 = torch.stack(mean_vals['step_epoch_log_acc2']).mean()
        assert epoch_end_metrics['step_epoch_log_and_pbar_acc1'] == eval_1
        assert epoch_end_metrics['step_epoch_log_acc2'] == eval_2
        assert 'step_epoch_pbar_acc3' not in epoch_end_metrics
        assert len(logged_metrics) == 3

    # make sure we are using the correct metrics for callbacks
    assert trainer.callback_metrics['early_stop_on'] == 171
    assert trainer.callback_metrics['checkpoint_on'] == 171

    # -------------------------------
    # VERIFY PBAR METRICS
    # -------------------------------
    # make sure pbar metrics are correct ang log metrics did not leak
    all_pbar_metrics = trainer.debug_pbar_added_metrics
    assert len(all_pbar_metrics) == (epochs * batches) + epochs

    epoch_idx = -1
    for i_start in range(0, len(all_pbar_metrics), batches + 1):
        epoch_idx += 1
        epoch_outputs = all_pbar_metrics[i_start: i_start + batches + 1]
        mean_vals = {
            'step_epoch_log_and_pbar_acc1': [],
            'step_epoch_pbar_acc3': []
        }

        # make sure each batch logged the expected value
        for batch_idx in range(len(epoch_outputs) - 1):
            logged_metrics = epoch_outputs[batch_idx]

            expected_val_1 = (5 + batch_idx) * (epoch_idx + 1)
            expected_val_2 = (7 + batch_idx) * (epoch_idx + 1)
            mean_vals['step_epoch_log_and_pbar_acc1'].append(torch.tensor(expected_val_1).float())
            mean_vals['step_epoch_pbar_acc3'].append(torch.tensor(expected_val_2).float())
            assert logged_metrics['step_epoch_log_and_pbar_acc1'] == expected_val_1
            assert logged_metrics['step_epoch_pbar_acc3'] == expected_val_2
            assert 'step_epoch_log_acc2' not in logged_metrics
            assert len(logged_metrics) == 3

        # make sure the metrics for the epoch end are actual means (the default reduce fx) or all the batches
        epoch_end_metrics = epoch_outputs[-1]
        eval_1 = torch.stack(mean_vals['step_epoch_log_and_pbar_acc1']).mean()
        eval_2 = torch.stack(mean_vals['step_epoch_pbar_acc3']).mean()
        assert epoch_end_metrics['step_epoch_log_and_pbar_acc1'] == eval_1
        assert epoch_end_metrics['step_epoch_pbar_acc3'] == eval_2
        assert 'step_epoch_log_acc2' not in epoch_end_metrics
        assert len(logged_metrics) == 3

    # -----------------------------------------
    # make sure training outputs what is expected
    # -----------------------------------------
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert len(out.batch_log_metrics) == 2

    train_step_out = out.training_step_output_for_epoch_end
    assert isinstance(train_step_out, TrainResult)

    assert 'minimize' in train_step_out
    assert f'step_epoch_log_and_pbar_acc1' in train_step_out
    assert f'step_epoch_log_acc2' in train_step_out

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.optimizer_closure(batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'] == (42.0 * 3) + (15.0 * 3)


def test_training_step_epoch_end_result(tmpdir):
    """
    Makes sure training_step and epoch_end can be used with Results (without batch_end)
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_log_epoch_and_step
    model.training_epoch_end = model.training_epoch_end_return_for_log_epoch_and_step
    model.val_dataloader = None

    batches = 3
    epochs = 1
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        row_log_interval=1,
        limit_train_batches=batches,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert model.training_epoch_end_called

    # make sure correct metrics were logged
    logged_metrics = trainer.debug_logged_metrics
    assert len(logged_metrics) == (epochs * batches) + epochs
    last_logged = logged_metrics[-1]

    assert last_logged['step_epoch_log_and_pbar_acc1'] == 210.0
    assert last_logged['step_epoch_log_acc2'] == 336.0
    assert last_logged['epoch_end_log_acc'] == 1212.0
    assert last_logged['epoch_end_log_pbar_acc'] == 1214.0
    assert 'epoch_end_pbar_acc' not in last_logged

    # make sure pbar metrics are correct
    logged_pbar = trainer.debug_pbar_added_metrics
    assert len(logged_pbar) == (epochs * batches) + epochs

    assert trainer.progress_bar_metrics['step_epoch_log_and_pbar_acc1'] == 210.0
    assert trainer.progress_bar_metrics['step_epoch_pbar_acc3'] == 504.0
    assert trainer.progress_bar_metrics['epoch_end_pbar_acc'] == 1213.0
    assert trainer.progress_bar_metrics['epoch_end_log_pbar_acc'] == 1214.0
    assert 'epoch_end_log_acc' not in trainer.progress_bar_metrics
    assert 'log_acc2' not in trainer.progress_bar_metrics

    # make sure callback metrics didn't change
    assert trainer.callback_metrics['early_stop_on'] == 171
    assert trainer.callback_metrics['checkpoint_on'] == 171

    # -----------------------------------------
    # make sure training outputs what is expected
    # -----------------------------------------
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert len(out.batch_log_metrics) == 2

    train_step_out = out.training_step_output_for_epoch_end
    assert isinstance(train_step_out, TrainResult)

    assert 'minimize' in train_step_out
    assert f'step_epoch_log_and_pbar_acc1' in train_step_out
    assert f'step_epoch_log_acc2' in train_step_out

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.optimizer_closure(batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'] == (42.0 * 3) + (15.0 * 3)
