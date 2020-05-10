from unittest.mock import patch, MagicMock

import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from tests.base import EvalModelTemplate


@patch('pytorch_lightning.loggers.neptune.neptune')
def test_neptune_online(neptune):
    logger = NeptuneLogger(api_key='test', project_name='project')

    created_experiment = neptune.Session.with_default_backend().get_project().create_experiment()

    # It's important to check if the internal variable _experiment was initialized in __init__.
    # Calling logger.experiment would cause a side-effect of initializing _experiment,
    # if it wasn't already initialized.
    assert logger._experiment == created_experiment
    assert logger.name == created_experiment.name
    assert logger.version == created_experiment.id


@patch('pytorch_lightning.loggers.neptune.neptune')
def test_neptune_offline(neptune):
    logger = NeptuneLogger(offline_mode=True)

    neptune.Session.assert_called_once_with(backend=neptune.OfflineBackend())
    assert logger.experiment == neptune.Session().get_project().create_experiment()


@patch('pytorch_lightning.loggers.neptune.neptune')
def test_neptune_additional_methods(neptune):
    logger = NeptuneLogger(api_key='test', project_name='project')

    created_experiment = neptune.Session.with_default_backend().get_project().create_experiment()

    logger.log_metric('test', torch.ones(1))
    created_experiment.log_metric.assert_called_once_with('test', torch.ones(1))
    created_experiment.log_metric.reset_mock()

    logger.log_metric('test', 1.0)
    created_experiment.log_metric.assert_called_once_with('test', 1.0)
    created_experiment.log_metric.reset_mock()

    logger.log_metric('test', 1.0, step=2)
    created_experiment.log_metric.assert_called_once_with('test', x=2, y=1.0)
    created_experiment.log_metric.reset_mock()

    logger.log_text('test', 'text')
    created_experiment.log_metric.assert_called_once_with('test', 'text')
    created_experiment.log_metric.reset_mock()

    logger.log_image('test', 'image file')
    created_experiment.log_image.assert_called_once_with('test', 'image file')
    created_experiment.log_image.reset_mock()

    logger.log_image('test', 'image file', step=2)
    created_experiment.log_image.assert_called_once_with('test', x=2, y='image file')
    created_experiment.log_image.reset_mock()

    logger.log_artifact('file')
    created_experiment.log_artifact.assert_called_once_with('file', None)

    logger.set_property('property', 10)
    created_experiment.set_property.assert_called_once_with('property', 10)

    logger.append_tags('one tag')
    created_experiment.append_tags.assert_called_once_with('one tag')
    created_experiment.append_tags.reset_mock()

    logger.append_tags(['two', 'tags'])
    created_experiment.append_tags.assert_called_once_with('two', 'tags')


def test_neptune_leave_open_experiment_after_fit(tmpdir):
    """Verify that neptune experiment was closed after training"""
    model = EvalModelTemplate()

    def _run_training(logger):
        logger._experiment = MagicMock()
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            train_percent_check=0.05,
            logger=logger
        )
        trainer.fit(model)
        return logger

    logger_close_after_fit = _run_training(NeptuneLogger(offline_mode=True))
    assert logger_close_after_fit._experiment.stop.call_count == 1

    logger_open_after_fit = _run_training(NeptuneLogger(offline_mode=True, close_after_fit=False))
    assert logger_open_after_fit._experiment.stop.call_count == 0
