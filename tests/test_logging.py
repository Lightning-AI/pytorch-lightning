import os.path
import shutil

import numpy as np
from pytorch_lightning import Trainer

from pytorch_lightning.testing import LightningTestModel

from .test_models import get_hparams, get_test_tube_logger, init_save_dir, clear_save_dir


def test_testtube_logger():
    """verify that basic functionality of test tube logger works"""

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    logger = get_test_tube_logger(False)
    logger.log_hyperparams(hparams)
    logger.save()

    trainer_options = dict(
        max_nb_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, "Training failed"

    clear_save_dir()


def test_mlflow_logger():
    """verify that basic functionality of mlflow logger works"""
    try:
        from pytorch_lightning.logging import MLFlowLogger
    except ModuleNotFoundError:
        return

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    mlflow_dir = os.path.join(root_dir, "mlruns")

    logger = MLFlowLogger("test", f"file://{mlflow_dir}")
    logger.log_hyperparams(hparams)
    logger.save()

    trainer_options = dict(
        max_nb_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, "Training failed"

    n = np.random.randint(0, 10000000, 1)[0]
    shutil.move(mlflow_dir, mlflow_dir + f'_{n}')
