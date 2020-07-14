"""
Tests to ensure that the training loop works with a dict
"""
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel

# train step + val step (no return)
# train step + val step (scalar return)
# train loop + val step (arbitrary dict return)
# train loop + val step (structured return)
# train loop + val step + val step end
# train loop + val step + val step end + val epoch end
# train loop + val step + val epoch end


def test_validation_step_no_return(tmpdir):
    """
    Test that val step can return nothing
    """
    model = DeterministicModel()
    model.training_step = model.training_step_dict_return
    model.validation_step = model.validation_step_no_return
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        weights_summary=None,
    )
    trainer.fit(model)

    # out are the results of the full loop
    # eval_results are output of _evaluate
    out, eval_results = trainer.run_evaluation(test_mode=False)
    assert len(out) == 0
    assert len(eval_results) == 0

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called


def test_validation_step_scalar_return(tmpdir):
    """
    Test that val step can return a scalar
    """
    model = DeterministicModel()
    model.training_step = model.training_step_dict_return
    model.validation_step = model.validation_step_scalar_return
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        weights_summary=None,
        limit_train_batches=2,
        limit_val_batches=2
    )
    trainer.fit(model)

    # out are the results of the full loop
    # eval_results are output of _evaluate
    out, eval_results = trainer.run_evaluation(test_mode=False)
    assert len(out) == 0
    assert len(eval_results) == 2
    assert eval_results[0] == 171 and eval_results[1] == 171

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called


def test_validation_step_arbitrary_dict_return(tmpdir):
    """
    Test that val step can return a scalar
    """
    model = DeterministicModel()
    model.training_step = model.training_step_dict_return
    model.validation_step = model.validation_step_arbitary_dict_return
    model.validation_step_end = None
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        weights_summary=None,
        limit_train_batches=2,
        limit_val_batches=2
    )
    trainer.fit(model)

    # out are the results of the full loop
    # eval_results are output of _evaluate
    callback_metrics, eval_results = trainer.run_evaluation(test_mode=False)
    assert len(callback_metrics) == 2
    assert len(eval_results) == 2
    assert eval_results[0]['some'] == 171
    assert eval_results[1]['some'] == 171

    assert eval_results[0]['value'] == 'a'
    assert eval_results[1]['value'] == 'a'

    # make sure correct steps were called
    assert model.validation_step_called
    assert not model.validation_step_end_called
    assert not model.validation_epoch_end_called

test_validation_step_arbitrary_dict_return('')


def training_step_with_step_end(tmpdir):
    """
    Checks train_step + training_step_end
    """
    model = DeterministicModel()
    model.training_step = model.training_step_for_step_end_dict
    model.training_step_end = model.training_step_end_dict
    model.val_dataloader = None

    trainer = Trainer(fast_dev_run=True, weights_summary=None)
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert model.training_step_end_called
    assert not model.training_epoch_end_called

    # make sure training outputs what is expected
    batch_idx, batch = 0, next(iter(model.train_dataloader()))

    out = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert out.batch_log_metrics['log_acc1'] == 14.0
    assert out.batch_log_metrics['log_acc2'] == 9.0

    train_step_end_out = out.training_step_output_for_epoch_end
    pbar_metrics = train_step_end_out['progress_bar']
    assert 'train_step_end' in train_step_end_out
    assert pbar_metrics['pbar_acc1'] == 19.0
    assert pbar_metrics['pbar_acc2'] == 21.0


def test_full_training_loop_dict(tmpdir):
    """
    Checks train_step + training_step_end + training_epoch_end
    """
    model = DeterministicModel()
    model.training_step = model.training_step_for_step_end_dict
    model.training_step_end = model.training_step_end_dict
    model.training_epoch_end = model.training_epoch_end_dict
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert model.training_step_end_called
    assert model.training_epoch_end_called

    # assert epoch end metrics were added
    assert trainer.callback_metrics['epoch_end_log_1'] == 178
    assert trainer.progress_bar_metrics['epoch_end_pbar_1'] == 234

    # make sure training outputs what is expected
    batch_idx, batch = 0, next(iter(model.train_dataloader()))

    out = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert out.batch_log_metrics['log_acc1'] == 14.0
    assert out.batch_log_metrics['log_acc2'] == 9.0

    train_step_end_out = out.training_step_output_for_epoch_end
    pbar_metrics = train_step_end_out['progress_bar']
    assert pbar_metrics['pbar_acc1'] == 19.0
    assert pbar_metrics['pbar_acc2'] == 21.0


def test_train_step_epoch_end(tmpdir):
    """
    Checks train_step + training_epoch_end (NO training_step_end)
    """
    model = DeterministicModel()
    model.training_step = model.training_step_dict_return
    model.training_step_end = None
    model.training_epoch_end = model.training_epoch_end_dict
    model.val_dataloader = None

    trainer = Trainer(max_epochs=1, weights_summary=None)
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert model.training_epoch_end_called

    # assert epoch end metrics were added
    assert trainer.callback_metrics['epoch_end_log_1'] == 178
    assert trainer.progress_bar_metrics['epoch_end_pbar_1'] == 234

    # make sure training outputs what is expected
    batch_idx, batch = 0, next(iter(model.train_dataloader()))

    out = trainer.run_training_batch(batch, batch_idx)
    assert out.signal == 0
    assert out.batch_log_metrics['log_acc1'] == 12.0
    assert out.batch_log_metrics['log_acc2'] == 7.0

    train_step_end_out = out.training_step_output_for_epoch_end
    pbar_metrics = train_step_end_out['progress_bar']
    assert pbar_metrics['pbar_acc1'] == 17.0
    assert pbar_metrics['pbar_acc2'] == 19.0
