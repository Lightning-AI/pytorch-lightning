"""
Tests to ensure that the training loop works with a dict
"""
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel


def test_training_step_result(tmpdir):
    """
    Tests that only training_step can be used
    """
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
    assert out.batch_log_metrics['log_acc1'] == 12.0
    assert out.batch_log_metrics['log_acc2'] == 7.0

    train_step_out = out.training_step_output_for_epoch_end
    pbar_metrics = train_step_out['progress_bar']
    assert 'log' in train_step_out
    assert 'progress_bar' in train_step_out
    assert train_step_out['train_step_test'] == 549
    assert pbar_metrics['pbar_acc1'] == 17.0
    assert pbar_metrics['pbar_acc2'] == 19.0

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.optimizer_closure(batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'] == (42.0 * 3) + (15.0 * 3)

test_training_step_result('')