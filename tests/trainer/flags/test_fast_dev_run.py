import pytest
from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate


def test_skip_on_fast_dev_run_batch_scaler(tmpdir):
    """ Test that batch scaler is skipped if fast dev run is enabled """

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        auto_scale_batch_size=True,
        fast_dev_run=True
    )
    expected_message = 'Skipping batch size scaler `fast_dev_run=True`'
    with pytest.warns(UserWarning, match=expected_message):
        trainer.tune(model)
