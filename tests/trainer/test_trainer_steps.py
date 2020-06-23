from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel


def test_trainingstep_dict(tmpdir):
    """
    Tests that only training_step can be used
    """
    model = DeterministicModel()
    model.training_step = model.training_step_dict_return
    model.val_dataloader = None

    trainer = Trainer(fast_dev_run=True, weights_summary=None)
    trainer.fit(model)

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.run_training_batch(batch, batch_idx)
    signal, grad_norm_dic, all_log_metrics, training_step_output_for_epoch_end = out
    assert signal == 0
    assert all_log_metrics['log_acc1'] == 12.0
    assert all_log_metrics['log_acc2'] == 7.0

    pbar_metrics = training_step_output_for_epoch_end['pbar_on_batch_end']
    assert pbar_metrics['pbar_acc1'] == 17.0
    assert pbar_metrics['pbar_acc2'] == 19.0
