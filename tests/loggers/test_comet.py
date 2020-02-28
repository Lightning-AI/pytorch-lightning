import os
import pickle

import tests.models.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import (
    CometLogger
)
from tests.models import LightningTestModel


def test_comet_logger(tmpdir, monkeypatch):
    """Verify that basic functionality of Comet.ml logger works."""

    # prevent comet logger from trying to print at exit, since
    # pytest's stdout/stderr redirection breaks it
    import atexit
    monkeypatch.setattr(atexit, "register", lambda _: None)

    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    comet_dir = os.path.join(tmpdir, "cometruns")

    # We test CometLogger in offline mode with local saves
    logger = CometLogger(
        save_dir=comet_dir,
        project_name="general",
        workspace="dummy-test",
    )

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print('result finished')
    assert result == 1, "Training failed"


def test_comet_pickle(tmpdir, monkeypatch):
    """Verify that pickling trainer with comet logger works."""

    # prevent comet logger from trying to print at exit, since
    # pytest's stdout/stderr redirection breaks it
    import atexit
    monkeypatch.setattr(atexit, "register", lambda _: None)

    tutils.reset_seed()

    # hparams = tutils.get_hparams()
    # model = LightningTestModel(hparams)

    comet_dir = os.path.join(tmpdir, "cometruns")

    # We test CometLogger in offline mode with local saves
    logger = CometLogger(
        save_dir=comet_dir,
        project_name="general",
        workspace="dummy-test",
    )

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})
