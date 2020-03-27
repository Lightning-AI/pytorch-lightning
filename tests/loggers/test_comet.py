import os
import pickle
from unittest.mock import patch

import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import LightningTestModel


def test_comet_logger(tmpdir, monkeypatch):
    """Verify that basic functionality of Comet.ml logger works."""

    # prevent comet logger from trying to print at exit, since
    # pytest's stdout/stderr redirection breaks it
    import atexit
    monkeypatch.setattr(atexit, 'register', lambda _: None)

    tutils.reset_seed()

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    comet_dir = os.path.join(tmpdir, 'cometruns')

    # We test CometLogger in offline mode with local saves
    logger = CometLogger(
        save_dir=comet_dir,
        project_name='general',
        workspace='dummy-test',
    )

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    trainer.logger.log_metrics({'acc': torch.ones(1)})

    assert result == 1, 'Training failed'


def test_comet_logger_online():
    """Test comet online with mocks."""
    # Test api_key given
    with patch('pytorch_lightning.loggers.comet.CometExperiment') as comet:
        logger = CometLogger(
            api_key='key',
            workspace='dummy-test',
            project_name='general'
        )

        _ = logger.experiment

        comet.assert_called_once_with(
            api_key='key',
            workspace='dummy-test',
            project_name='general'
        )

    # Test both given
    with patch('pytorch_lightning.loggers.comet.CometExperiment') as comet:
        logger = CometLogger(
            save_dir='test',
            api_key='key',
            workspace='dummy-test',
            project_name='general'
        )

        _ = logger.experiment

        comet.assert_called_once_with(
            api_key='key',
            workspace='dummy-test',
            project_name='general'
        )

    # Test neither given
    with pytest.raises(MisconfigurationException):
        CometLogger(
            workspace='dummy-test',
            project_name='general'
        )

    # Test already exists
    with patch('pytorch_lightning.loggers.comet.CometExistingExperiment') as comet_existing:
        logger = CometLogger(
            experiment_key='test',
            experiment_name='experiment',
            api_key='key',
            workspace='dummy-test',
            project_name='general'
        )

        _ = logger.experiment

        comet_existing.assert_called_once_with(
            api_key='key',
            workspace='dummy-test',
            project_name='general',
            previous_experiment='test'
        )

        comet_existing().set_name.assert_called_once_with('experiment')

    with patch('pytorch_lightning.loggers.comet.API') as api:
        CometLogger(
            api_key='key',
            workspace='dummy-test',
            project_name='general',
            rest_api_key='rest'
        )

        api.assert_called_once_with('rest')


def test_comet_pickle(tmpdir, monkeypatch):
    """Verify that pickling trainer with comet logger works."""

    # prevent comet logger from trying to print at exit, since
    # pytest's stdout/stderr redirection breaks it
    import atexit
    monkeypatch.setattr(atexit, 'register', lambda _: None)

    tutils.reset_seed()

    # hparams = tutils.get_default_hparams()
    # model = LightningTestModel(hparams)

    comet_dir = os.path.join(tmpdir, 'cometruns')

    # We test CometLogger in offline mode with local saves
    logger = CometLogger(
        save_dir=comet_dir,
        project_name='general',
        workspace='dummy-test',
    )

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({'acc': 1.0})
