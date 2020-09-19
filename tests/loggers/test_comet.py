import os
from unittest.mock import patch

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


def test_comet_logger_online():
    """Test comet online with mocks."""
    # Test api_key given
    with patch('pytorch_lightning.loggers.comet.CometExperiment') as comet:
        logger = CometLogger(api_key='key', workspace='dummy-test', project_name='general')

        _ = logger.experiment

        comet.assert_called_once_with(api_key='key', workspace='dummy-test', project_name='general')

    # Test both given
    with patch('pytorch_lightning.loggers.comet.CometExperiment') as comet:
        logger = CometLogger(save_dir='test', api_key='key', workspace='dummy-test', project_name='general')

        _ = logger.experiment

        comet.assert_called_once_with(api_key='key', workspace='dummy-test', project_name='general')

    # Test neither given
    with pytest.raises(MisconfigurationException):
        CometLogger(workspace='dummy-test', project_name='general')

    # Test already exists
    with patch('pytorch_lightning.loggers.comet.CometExistingExperiment') as comet_existing:
        logger = CometLogger(
            experiment_key='test',
            experiment_name='experiment',
            api_key='key',
            workspace='dummy-test',
            project_name='general',
        )

        _ = logger.experiment

        comet_existing.assert_called_once_with(
            api_key='key', workspace='dummy-test', project_name='general', previous_experiment='test'
        )

        comet_existing().set_name.assert_called_once_with('experiment')

    with patch('pytorch_lightning.loggers.comet.API') as api:
        CometLogger(api_key='key', workspace='dummy-test', project_name='general', rest_api_key='rest')

        api.assert_called_once_with('rest')


def test_comet_logger_experiment_name():
    """Test that Comet Logger experiment name works correctly."""

    api_key = "key"
    experiment_name = "My Name"

    # Test api_key given
    with patch('pytorch_lightning.loggers.comet.CometExperiment') as comet:
        logger = CometLogger(api_key=api_key, experiment_name=experiment_name,)

        assert logger._experiment is None

        _ = logger.experiment

        comet.assert_called_once_with(api_key=api_key, project_name=None)

        comet().set_name.assert_called_once_with(experiment_name)


def test_comet_logger_dirs_creation(tmpdir, monkeypatch):
    """ Test that the logger creates the folders and files in the right place. """
    # prevent comet logger from trying to print at exit, since
    # pytest's stdout/stderr redirection breaks it
    import atexit

    monkeypatch.setattr(atexit, 'register', lambda _: None)

    logger = CometLogger(project_name='test', save_dir=tmpdir)
    assert not os.listdir(tmpdir)
    assert logger.mode == 'offline'
    assert logger.save_dir == tmpdir

    _ = logger.experiment
    version = logger.version
    assert set(os.listdir(tmpdir)) == {f'{logger.experiment.id}.zip'}

    model = EvalModelTemplate()
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=1, limit_val_batches=3)
    trainer.fit(model)

    assert trainer.checkpoint_callback.dirpath == (tmpdir / 'test' / version / 'checkpoints')
    assert set(os.listdir(trainer.checkpoint_callback.dirpath)) == {'epoch=0.ckpt'}


def test_comet_name_default():
    """ Test that CometLogger.name don't create an Experiment and returns a default value. """

    api_key = "key"

    with patch('pytorch_lightning.loggers.comet.CometExperiment') as comet:
        logger = CometLogger(api_key=api_key)

        assert logger._experiment is None

        assert logger.name == "comet-default"

        assert logger._experiment is None


def test_comet_name_project_name():
    """ Test that CometLogger.name does not create an Experiment and returns project name if passed. """

    api_key = "key"
    project_name = "My Project Name"

    with patch('pytorch_lightning.loggers.comet.CometExperiment') as comet:
        logger = CometLogger(api_key=api_key, project_name=project_name)

        assert logger._experiment is None

        assert logger.name == project_name

        assert logger._experiment is None


def test_comet_version_without_experiment():
    """ Test that CometLogger.version does not create an Experiment. """

    api_key = "key"
    experiment_name = "My Name"

    with patch('pytorch_lightning.loggers.comet.CometExperiment') as comet:
        logger = CometLogger(api_key=api_key, experiment_name=experiment_name)

        assert logger._experiment is None

        first_version = logger.version
        assert first_version is not None

        assert logger.version == first_version

        assert logger._experiment is None

        _ = logger.experiment

        logger.reset_experiment()

        second_version = logger.version
        assert second_version is not None
        assert second_version != first_version


def test_comet_epoch_logging(tmpdir, monkeypatch):
    """ Test that CometLogger removes the epoch key from the metrics dict and passes it as argument. """
    import atexit

    monkeypatch.setattr(atexit, "register", lambda _: None)
    with patch("pytorch_lightning.loggers.comet.CometOfflineExperiment.log_metrics") as log_metrics:
        logger = CometLogger(project_name="test", save_dir=tmpdir)
        logger.log_metrics({"test": 1, "epoch": 1}, step=123)
        log_metrics.assert_called_once_with({"test": 1}, epoch=1, step=123)
