import os
import pickle

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from tests.base import LightningTestModel


def test_mlflow_logger_exists(tmpdir):
    """Verify that basic functionality of mlflow logger works."""
    logger = MLFlowLogger('test', save_dir=tmpdir)
    # Test already exists
    logger2 = MLFlowLogger('test', save_dir=tmpdir)
    assert logger2.run_id


def test_mlflow_pickle(tmpdir):
    """Verify that pickling trainer with mlflow logger works."""
    tutils.reset_seed()

    mlflow_dir = os.path.join(tmpdir, 'mlruns')
    logger = MLFlowLogger('test', save_dir=mlflow_dir)
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({'acc': 1.0})
