import os
import pickle

import pytest
import torch

import tests.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.logging import (
    LightningLoggerBase,
    rank_zero_only,
    TensorBoardLogger,
)
from pytorch_lightning.testing import LightningTestModel


def test_testtube_logger(tmpdir):
    """Verify that basic functionality of test tube logger works."""
    tutils.reset_seed()
    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    logger = tutils.get_test_tube_logger(tmpdir, False)

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.01,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, "Training failed"


def test_testtube_pickle(tmpdir):
    """Verify that pickling a trainer containing a test tube logger works."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    logger = tutils.get_test_tube_logger(tmpdir, False)
    logger.log_hyperparams(hparams)
    logger.save()

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.01,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_mlflow_logger(tmpdir):
    """Verify that basic functionality of mlflow logger works."""
    tutils.reset_seed()

    try:
        from pytorch_lightning.logging import MLFlowLogger
    except ModuleNotFoundError:
        return

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    mlflow_dir = os.path.join(tmpdir, "mlruns")
    logger = MLFlowLogger("test", tracking_uri=f"file:{os.sep * 2}{mlflow_dir}")

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.01,
        logger=logger
    )
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print('result finished')
    assert result == 1, "Training failed"


def test_mlflow_pickle(tmpdir):
    """Verify that pickling trainer with mlflow logger works."""
    tutils.reset_seed()

    try:
        from pytorch_lightning.logging import MLFlowLogger
    except ModuleNotFoundError:
        return

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


def test_comet_logger(tmpdir, monkeypatch):
    """Verify that basic functionality of Comet.ml logger works."""

    # prevent comet logger from trying to print at exit, since
    # pytest's stdout/stderr redirection breaks it
    import atexit
    monkeypatch.setattr(atexit, "register", lambda _: None)

    tutils.reset_seed()

    try:
        from pytorch_lightning.logging import CometLogger
    except ModuleNotFoundError:
        return

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
        train_percent_check=0.01,
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

    try:
        from pytorch_lightning.logging import CometLogger
    except ModuleNotFoundError:
        return

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


def test_wandb_logger(tmpdir):
    """Verify that basic functionality of wandb logger works."""
    tutils.reset_seed()

    from pytorch_lightning.logging import WandbLogger

    wandb_dir = os.path.join(tmpdir, "wandb")
    logger = WandbLogger(save_dir=wandb_dir, anonymous=True)


def test_neptune_logger(tmpdir):
    """Verify that basic functionality of neptune logger works."""
    tutils.reset_seed()

    from pytorch_lightning.logging import NeptuneLogger

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)
    logger = NeptuneLogger(offline_mode=True)

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.01,
        logger=logger
    )
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print('result finished')
    assert result == 1, "Training failed"


def test_wandb_pickle(tmpdir):
    """Verify that pickling trainer with wandb logger works."""
    tutils.reset_seed()

    from pytorch_lightning.logging import WandbLogger
    wandb_dir = str(tmpdir)
    logger = WandbLogger(save_dir=wandb_dir, anonymous=True)
    assert logger is not None


def test_neptune_pickle(tmpdir):
    """Verify that pickling trainer with neptune logger works."""
    tutils.reset_seed()

    from pytorch_lightning.logging import NeptuneLogger

    # hparams = tutils.get_hparams()
    # model = LightningTestModel(hparams)

    logger = NeptuneLogger(offline_mode=True)

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_tensorboard_logger(tmpdir):
    """Verify that basic functionality of Tensorboard logger works."""

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    logger = TensorBoardLogger(save_dir=tmpdir, name="tensorboard_logger_test")

    trainer_options = dict(max_epochs=1, train_percent_check=0.01, logger=logger)

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print("result finished")
    assert result == 1, "Training failed"


def test_tensorboard_pickle(tmpdir):
    """Verify that pickling trainer with Tensorboard logger works."""

    # hparams = tutils.get_hparams()
    # model = LightningTestModel(hparams)

    logger = TensorBoardLogger(save_dir=tmpdir, name="tensorboard_pickle_test")

    trainer_options = dict(max_epochs=1, logger=logger)

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})


def test_tensorboard_automatic_versioning(tmpdir):
    """Verify that automatic versioning works"""

    root_dir = tmpdir.mkdir("tb_versioning")
    root_dir.mkdir("0")
    root_dir.mkdir("1")

    logger = TensorBoardLogger(save_dir=tmpdir, name="tb_versioning")

    assert logger.version == 2


def test_tensorboard_manual_versioning(tmpdir):
    """Verify that manual versioning works"""

    root_dir = tmpdir.mkdir("tb_versioning")
    root_dir.mkdir("0")
    root_dir.mkdir("1")
    root_dir.mkdir("2")

    logger = TensorBoardLogger(save_dir=tmpdir, name="tb_versioning", version=1)

    assert logger.version == 1


@pytest.mark.parametrize("step_idx", [10, None])
def test_tensorboard_log_metrics(tmpdir, step_idx):
    logger = TensorBoardLogger(tmpdir)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatTensor": torch.tensor(0.1),
        "IntTensor": torch.tensor(1)
    }
    logger.log_metrics(metrics, step_idx)


def test_tensorboard_log_hyperparams(tmpdir):
    logger = TensorBoardLogger(tmpdir)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True
    }
    logger.log_hyperparams(hparams)


def test_custom_logger(tmpdir):
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
        def log_metrics(self, metrics, step):
            self.metrics_logged = metrics

        @rank_zero_only
        def finalize(self, status):
            self.finalized_status = status

        @property
        def name(self):
            return "name"

        @property
        def version(self):
            return "1"

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    logger = CustomLogger()

    trainer_options = dict(
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger,
        default_save_path=tmpdir
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    assert result == 1, "Training failed"
    assert logger.hparams_logged == hparams
    assert logger.metrics_logged != {}
    assert logger.finalized_status == "success"


def test_sacred_logger(tmpdir):
    """Verify that basic functionality of sacred logger works."""
    tutils.reset_seed()

    try:
        from pytorch_lightning.logging import SacredLogger
    except ModuleNotFoundError:
        return

    try:
        from sacred import Experiment
    except ModuleNotFoundError:
        return

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    mlflow_dir = os.path.join(tmpdir, "sacredruns")
    ex = Experiment()
    ex_config = vars(hparams)
    ex.add_config(ex_config)
    # ex.observers.append(
    #     MongoObserver.create(
    #         url='mongodb://{ip}:{port}'.format(**mongodb_settings),
    #         db_name='{db}'.format(**mongodb_settings),
    #     )
    # )


    def run_fct():
        logger = SacredLogger(ex)

        trainer_options = dict(
            default_save_path=tmpdir,
            max_epochs=1,
            train_percent_check=0.01,
            logger=logger
        )
        trainer = Trainer(**trainer_options)
        result = trainer.fit(model)
        return result


    result = ex.run(run_fct) # TODO: will this return the result?

    print('result finished')
    assert result == 1, "Training failed"
