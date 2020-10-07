import pytest
from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate

@pytest.mark.parametrize('tuner_alg', ['scale_batch_size', 'lr_find'])
def test_skip_on_fast_dev_run_batch_scaler(tmpdir, tuner_alg):
    """ Test that batch scaler is skipped if fast dev run is enabled """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        auto_scale_batch_size=True if tuner_alg=='scale_batch_size' else False,
        auto_lr_find=True if tuner_alg=='lr_find' else False,
        fast_dev_run=True
    )
    alg = 'batch size scaler' if tuner_alg=='scale_batch_size' else 'learning rate finder'
    expected_message = f'Skipping {alg} since `fast_dev_run=True`'
    with pytest.warns(UserWarning, match=expected_message):
        trainer.tune(model)
