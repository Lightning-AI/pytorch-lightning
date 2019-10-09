import os.path
import pickle
import shutil

import numpy as np
import torch
from pytorch_lightning import Trainer

from pytorch_lightning.testing import LightningTestModel
from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only
from .test_models import get_hparams, get_test_tube_logger, init_save_dir, clear_save_dir

RANDOM_SEEDS = list(np.random.randint(0, 10000, 1000))


def test_testtube_logger():
    """
    verify that basic functionality of test tube logger works
    """
    reset_seed()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    logger = get_test_tube_logger(False)

    trainer_options = dict(
        max_nb_epochs=1,
        train_percent_check=0.01,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, "Training failed"

    clear_save_dir()


def test_testtube_pickle():
    """
    Verify that pickling a trainer containing a test tube logger works
    """
    reset_seed()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    logger = get_test_tube_logger(False)
    logger.log_hyperparams(hparams)
    logger.save()

    trainer_options = dict(
        max_nb_epochs=1,
        train_percent_check=0.01,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_mlflow_logger():
    """
    verify that basic functionality of mlflow logger works
    """
    reset_seed()

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
        train_percent_check=0.01,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, "Training failed"

    n = np.random.randint(0, 10000000, 1)[0]
    shutil.move(mlflow_dir, mlflow_dir + f'_{n}')


def test_mlflow_pickle():
    """
    verify that pickling trainer with mlflow logger works
    """
    reset_seed()

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
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_custom_logger():

    class CustomLogger(LightningLoggerBase):
        def __init__(self):
            super().__init__()
            self.hparams_logged = None
            self.metrics_logged = None
            self.finalized = False

        @rank_zero_only
        def log_hyperparams(self, params):
            self.hparams_logged = params

        @rank_zero_only
        def log_metrics(self, metrics, step_num):
            self.metrics_logged = metrics

        @rank_zero_only
        def finalize(self, status):
            self.finalized_status = status

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    logger = CustomLogger()

    trainer_options = dict(
        max_nb_epochs=1,
        train_percent_check=0.01,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    assert result == 1, "Training failed"
    assert logger.hparams_logged == hparams
    assert logger.metrics_logged != {}
    assert logger.finalized_status == "success"


def reset_seed():
    SEED = RANDOM_SEEDS.pop()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
