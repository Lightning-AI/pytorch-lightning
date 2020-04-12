import pickle

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from tests.base import LightningTestModel


def test_testtube_pickle(tmpdir):
    """Verify that pickling a trainer containing a test tube logger works."""
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()

    logger = TestTubeLogger(tmpdir, name='lightning_logs', debug=False, version=None)
    logger.log_hyperparams(hparams)
    logger.save()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({'acc': 1.0})
