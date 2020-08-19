"""
Tests to ensure that the training loop works with a dict
"""
import os

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.step_result import TrainResult
from tests.base import EvalModelTemplate
from tests.base.deterministic_model import DeterministicModel


# test with train_step_end
# add logging + row interval tests

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
    assert len(trainer.dev_debugger.logged_metrics) == batches
    for batch_idx, logged_metrics in enumerate(trainer.dev_debugger.logged_metrics):
        assert logged_metrics[f'step_log_and_pbar_acc1_b{batch_idx}'] == 11.0
        assert logged_metrics[f'step_log_acc2_b{batch_idx}'] == 12.0
        assert f'step_pbar_acc3_b{batch_idx}' not in logged_metrics
        assert len(logged_metrics) == 4

    # make sure we are using the correct metrics for callbacks
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
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
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
    assert len(trainer.dev_debugger.logged_metrics) == epochs
    epoch_metrics = trainer.dev_debugger.logged_metrics
    assert len(epoch_metrics) == epochs
    for batch_idx, logged_metrics in enumerate(epoch_metrics):
        assert logged_metrics[f'epoch_log_and_pbar_acc1_e{batch_idx}'] == 14.0
        assert logged_metrics[f'epoch_log_acc2_e{batch_idx}'] == 15.0
        assert f'epoch_pbar_acc3_e{batch_idx}' not in logged_metrics
        assert len(logged_metrics) == 4

    # make sure we are using the correct metrics for callbacks
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
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
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
    assert len(trainer.dev_debugger.logged_metrics) == (epochs * batches) + epochs
    epoch_metrics = trainer.dev_debugger.logged_metrics
    epoch_idx = -1
    for i_start in range(0, len(epoch_metrics), batches + 1):
        epoch_idx += 1
        epoch_outputs = epoch_metrics[i_start: i_start + batches + 1]
        mean_vals = {
            'epoch_step_epoch_log_and_pbar_acc1': [],
            'epoch_step_epoch_log_acc2': []
        }

        # make sure each batch logged the expected value
        for batch_idx in range(len(epoch_outputs) - 1):
            logged_metrics = epoch_outputs[batch_idx]

            expected_val_1 = (5 + batch_idx) * (epoch_idx + 1)
            expected_val_2 = (6 + batch_idx) * (epoch_idx + 1)
            mean_vals['epoch_step_epoch_log_and_pbar_acc1'].append(torch.tensor(expected_val_1).float())
            mean_vals['epoch_step_epoch_log_acc2'].append(torch.tensor(expected_val_2).float())

            assert logged_metrics['step_step_epoch_log_and_pbar_acc1'] == expected_val_1
            assert logged_metrics['step_step_epoch_log_acc2'] == expected_val_2
            assert 'step_epoch_pbar_acc3' not in logged_metrics
            assert len(logged_metrics) == 4

        # make sure the metrics for the epoch end are actual means (the default reduce fx) or all the batches
        epoch_end_metrics = epoch_outputs[-1]
        eval_1 = torch.stack(mean_vals['epoch_step_epoch_log_and_pbar_acc1']).mean()
        eval_2 = torch.stack(mean_vals['epoch_step_epoch_log_acc2']).mean()
        assert epoch_end_metrics['epoch_step_epoch_log_and_pbar_acc1'] == eval_1
        assert epoch_end_metrics['epoch_step_epoch_log_acc2'] == eval_2
        assert 'step_epoch_pbar_acc3' not in epoch_end_metrics
        assert len(logged_metrics) == 4

    # make sure we are using the correct metrics for callbacks
    assert trainer.callback_metrics['checkpoint_on'] == 171

    # -------------------------------
    # VERIFY PBAR METRICS
    # -------------------------------
    # make sure pbar metrics are correct ang log metrics did not leak
    all_pbar_metrics = trainer.dev_debugger.pbar_added_metrics
    assert len(all_pbar_metrics) == (epochs * batches) + epochs

    epoch_idx = -1
    for i_start in range(0, len(all_pbar_metrics), batches + 1):
        epoch_idx += 1
        epoch_outputs = all_pbar_metrics[i_start: i_start + batches + 1]
        mean_vals = {
            'epoch_step_epoch_log_and_pbar_acc1': [],
            'epoch_step_epoch_pbar_acc3': []
        }

        # make sure each batch logged the expected value
        for batch_idx in range(len(epoch_outputs) - 1):
            logged_metrics = epoch_outputs[batch_idx]

            expected_val_1 = (5 + batch_idx) * (epoch_idx + 1)
            expected_val_2 = (7 + batch_idx) * (epoch_idx + 1)
            mean_vals['epoch_step_epoch_log_and_pbar_acc1'].append(torch.tensor(expected_val_1).float())
            mean_vals['epoch_step_epoch_pbar_acc3'].append(torch.tensor(expected_val_2).float())
            assert logged_metrics['step_step_epoch_log_and_pbar_acc1'] == expected_val_1
            assert logged_metrics['step_step_epoch_pbar_acc3'] == expected_val_2
            assert 'step_epoch_log_acc2' not in logged_metrics
            assert len(logged_metrics) == 3

        # make sure the metrics for the epoch end are actual means (the default reduce fx) or all the batches
        epoch_end_metrics = epoch_outputs[-1]
        eval_1 = torch.stack(mean_vals['epoch_step_epoch_log_and_pbar_acc1']).mean()
        eval_2 = torch.stack(mean_vals['epoch_step_epoch_pbar_acc3']).mean()
        assert epoch_end_metrics['epoch_step_epoch_log_and_pbar_acc1'] == eval_1
        assert epoch_end_metrics['epoch_step_epoch_pbar_acc3'] == eval_2
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
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out, TrainResult)

    assert 'minimize' in train_step_out
    assert 'step_step_epoch_log_and_pbar_acc1' in train_step_out
    assert 'step_step_epoch_log_acc2' in train_step_out
    assert 'epoch_step_epoch_log_and_pbar_acc1' in train_step_out
    assert 'epoch_step_epoch_log_acc2' in train_step_out

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
    logged_metrics = trainer.dev_debugger.logged_metrics
    assert len(logged_metrics) == (epochs * batches) + epochs
    last_logged = logged_metrics[-1]

    assert last_logged['epoch_step_epoch_log_and_pbar_acc1'] == 210.0
    assert last_logged['epoch_step_epoch_log_acc2'] == 336.0
    assert last_logged['epoch_epoch_end_log_acc'] == 1212.0
    assert last_logged['epoch_epoch_end_log_pbar_acc'] == 1214.0
    assert 'epoch_end_pbar_acc' not in last_logged

    # make sure pbar metrics are correct
    logged_pbar = trainer.dev_debugger.pbar_added_metrics
    assert len(logged_pbar) == (epochs * batches) + epochs

    assert trainer.progress_bar_metrics['epoch_step_epoch_log_and_pbar_acc1'] == 210.0
    assert trainer.progress_bar_metrics['step_step_epoch_log_and_pbar_acc1'] == 7.0
    assert trainer.progress_bar_metrics['epoch_step_epoch_pbar_acc3'] == 504.0
    assert trainer.progress_bar_metrics['epoch_epoch_end_pbar_acc'] == 1213.0
    assert trainer.progress_bar_metrics['epoch_epoch_end_log_pbar_acc'] == 1214.0
    assert 'epoch_end_log_acc' not in trainer.progress_bar_metrics
    assert 'log_acc2' not in trainer.progress_bar_metrics

    # make sure callback metrics didn't change
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
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out, TrainResult)

    assert 'minimize' in train_step_out
    assert 'step_step_epoch_log_and_pbar_acc1' in train_step_out
    assert 'epoch_step_epoch_log_and_pbar_acc1' in train_step_out
    assert 'step_step_epoch_log_acc2' in train_step_out
    assert 'epoch_step_epoch_log_acc2' in train_step_out

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.optimizer_closure(batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'] == (42.0 * 3) + (15.0 * 3)


def test_no_auto_callbacks_with_train_loop_only(tmpdir):
    """
    Make sure early stop + checkpoint work with only a train loop
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_no_default_callbacks_for_train_loop
    model.training_epoch_end = None
    model.val_dataloader = None

    batches = 3
    epochs = 3
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        row_log_interval=1,
        limit_train_batches=batches,
        weights_summary=None,
    )
    trainer.fit(model)

    all_losses = trainer.dev_debugger.saved_train_losses
    assert len(all_losses) == batches * epochs

    assert trainer.checkpoint_callback.monitor == 'checkpoint_on'
    assert trainer.early_stop_callback is None

    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=True,
        max_epochs=epochs,
        row_log_interval=1,
        limit_train_batches=batches,
        weights_summary=None,
    )
    trainer.fit(model)

    assert trainer.early_stop_callback.monitor == 'val_loss'


def test_no_callbacks_with_train_loop_only(tmpdir):
    """
    Make sure early stop + checkpoint work with only a train loop
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_no_callbacks_result_obj
    model.training_epoch_end = None
    model.val_dataloader = None

    batches = 3
    epochs = 3
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        row_log_interval=1,
        limit_train_batches=batches,
        weights_summary=None,
    )
    trainer.fit(model)

    all_losses = trainer.dev_debugger.saved_train_losses
    assert len(all_losses) == batches * epochs

    assert trainer.early_stop_callback is None

    assert len(trainer.dev_debugger.checkpoint_callback_history) == 0
    assert len(trainer.dev_debugger.early_stopping_history) == 0


def test_use_callbacks_with_train_loop_only(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'

    model = DeterministicModel()
    model.training_step = model.training_step_result_log_epoch_and_step_for_callbacks
    model.training_epoch_end = None
    model.val_dataloader = None

    batches = 3
    epochs = 300
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        early_stop_callback=True,
        row_log_interval=1,
        limit_train_batches=batches,
        weights_summary=None,
    )
    trainer.fit(model)

    num_expected_epochs = 10

    # ----------------------------------
    # VERIFY EARLY STOPPING BEHAVIOR
    # ----------------------------------
    # with train loop only it happens on every epoch
    early_stop_vals = trainer.dev_debugger.early_stopping_history
    assert len(early_stop_vals) == num_expected_epochs
    min_val = min([x['best'] for x in early_stop_vals])
    assert min_val == 171 + 9
    all_losses = trainer.dev_debugger.saved_train_losses

    from collections import Counter
    batch_idxs = Counter([x['batch_idx'] for x in all_losses])
    for i, val in batch_idxs.items():
        assert val == num_expected_epochs
        assert i in [0, 1, 2]

    # ----------------------------------
    # VERIFY CHECKPOINTING BEHAVIOR
    # ----------------------------------
    ckpt_vals = trainer.dev_debugger.checkpoint_callback_history
    assert len(ckpt_vals) == 5, '5 ckpts should have been saved'
    for ckpt_val, expected_epoch in zip(ckpt_vals, [0, 1, 2, 3, 6]):
        assert ckpt_val['epoch'] == expected_epoch
        assert ckpt_val['monitor'] == 'checkpoint_on'


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_full_train_loop_with_results_obj_dp(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'

    batches = 10
    epochs = 3

    model = EvalModelTemplate()
    model.validation_step = None
    model.test_step = None
    model.training_step = model.training_step_full_loop_result_obj_dp
    model.training_step_end = model.training_step_end_full_loop_result_obj_dp
    model.training_epoch_end = model.training_epoch_end_full_loop_result_obj_dp
    model.val_dataloader = None
    model.test_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        distributed_backend='dp',
        gpus=[0, 1],
        max_epochs=epochs,
        early_stop_callback=True,
        row_log_interval=2,
        limit_train_batches=batches,
        weights_summary=None,
    )

    trainer.fit(model)

    # make sure we saw all the correct keys
    seen_keys = set()
    for metric in trainer.dev_debugger.logged_metrics:
        seen_keys.update(metric.keys())

    assert 'train_step_metric' in seen_keys
    assert 'train_step_end_metric' in seen_keys
    assert 'epoch_train_epoch_end_metric' in seen_keys


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_loop_steps_only_dp(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'

    batches = 10
    epochs = 3

    model = EvalModelTemplate()
    model.validation_step = None
    model.test_step = None
    model.training_step = model.training_step_result_obj_dp
    model.training_step_end = None
    model.training_epoch_end = None
    model.validation_step = model.validation_step_result_obj_dp
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        distributed_backend='dp',
        gpus=[0, 1],
        max_epochs=epochs,
        early_stop_callback=True,
        row_log_interval=2,
        limit_train_batches=batches,
        weights_summary=None,
    )

    trainer.fit(model)

    assert model.training_step_called
    assert model.validation_step_called


def test_result_map(tmpdir):
    result = TrainResult()
    result.log_dict({'x1': torch.tensor(1), 'x2': torch.tensor(2)})
    result.rename_keys({'x1': 'y1', 'x2': 'y2'})

    assert 'x1' not in result
    assert 'x2' not in result
    assert 'y1' in result
    assert 'y2' in result


def test_result_monitor_warnings(tmpdir):
    """
    Tests that we warn when the monitor key is changed and we use Results obj
    """
    model = EvalModelTemplate()
    model.test_step = None
    model.training_step = model.training_step_result_obj
    model.training_step_end = None
    model.training_epoch_end = None
    model.validation_step = model.validation_step_result_obj
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        early_stop_callback=True,
        row_log_interval=2,
        limit_train_batches=2,
        weights_summary=None,
        checkpoint_callback=ModelCheckpoint(monitor='not_val_loss')
    )

    with pytest.warns(UserWarning, match='key of ModelCheckpoint has no effect'):
        trainer.fit(model)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        row_log_interval=2,
        limit_train_batches=2,
        weights_summary=None,
        early_stop_callback=EarlyStopping(monitor='not_val_loss')
    )

    with pytest.warns(UserWarning, match='key of EarlyStopping has no effec'):
        trainer.fit(model)
