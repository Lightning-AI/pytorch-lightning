import pytest
from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate


@pytest.mark.parametrize('tuner_alg', ['batch size scaler', 'learning rate finder'])
def test_skip_on_fast_dev_run_tuner(tmpdir, tuner_alg):
    """ Test that tuner algorithms are skipped if fast dev run is enabled """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        auto_scale_batch_size=True if tuner_alg == 'batch size scaler' else False,
        auto_lr_find=True if tuner_alg == 'learning rate finder' else False,
        fast_dev_run=True
    )
    expected_message = f'Skipping {tuner_alg} since fast_dev_run is enabled.'
    with pytest.warns(UserWarning, match=expected_message):
        trainer.tune(model)
