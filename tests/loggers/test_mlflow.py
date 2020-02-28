import os
import pickle

import tests.models.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import (
    MLFlowLogger
)
from tests.models import LightningTestModel


def test_mlflow_logger(tmpdir):
    """Verify that basic functionality of mlflow logger works."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    mlflow_dir = os.path.join(tmpdir, "mlruns")
    logger = MLFlowLogger("test", tracking_uri=f"file:{os.sep * 2}{mlflow_dir}")

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


def test_mlflow_pickle(tmpdir):
    """Verify that pickling trainer with mlflow logger works."""
    tutils.reset_seed()

    # hparams = tutils.get_hparams()
    # model = LightningTestModel(hparams)

    mlflow_dir = os.path.join(tmpdir, "mlruns")
    logger = MLFlowLogger("test", tracking_uri=f"file:{os.sep * 2}{mlflow_dir}")
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})
