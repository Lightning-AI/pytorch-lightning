from unittest.mock import patch, MagicMock

import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from tests.base import LightningTestModel


@patch('pytorch_lightning.loggers.neptune.neptune')
def test_neptune_online(neptune):
    logger = NeptuneLogger(api_key='test', offline_mode=False, project_name='project')
    neptune.init.assert_called_once_with(api_token='test', project_qualified_name='project')

    assert logger.name == neptune.create_experiment().name
    assert logger.version == neptune.create_experiment().id


@patch('pytorch_lightning.loggers.neptune.neptune')
def test_neptune_additional_methods(neptune):
    logger = NeptuneLogger(offline_mode=True)

    logger.log_metric('test', torch.ones(1))
    neptune.create_experiment().log_metric.assert_called_once_with('test', torch.ones(1))
    neptune.create_experiment().log_metric.reset_mock()

    logger.log_metric('test', 1.0)
    neptune.create_experiment().log_metric.assert_called_once_with('test', 1.0)
    neptune.create_experiment().log_metric.reset_mock()

    logger.log_metric('test', 1.0, step=2)
    neptune.create_experiment().log_metric.assert_called_once_with('test', x=2, y=1.0)
    neptune.create_experiment().log_metric.reset_mock()

    logger.log_text('test', 'text')
    neptune.create_experiment().log_metric.assert_called_once_with('test', 'text')
    neptune.create_experiment().log_metric.reset_mock()

    logger.log_image('test', 'image file')
    neptune.create_experiment().log_image.assert_called_once_with('test', 'image file')
    neptune.create_experiment().log_image.reset_mock()

    logger.log_image('test', 'image file', step=2)
    neptune.create_experiment().log_image.assert_called_once_with('test', x=2, y='image file')
    neptune.create_experiment().log_image.reset_mock()

    logger.log_artifact('file')
    neptune.create_experiment().log_artifact.assert_called_once_with('file', None)

    logger.set_property('property', 10)
    neptune.create_experiment().set_property.assert_called_once_with('property', 10)

    logger.append_tags('one tag')
    neptune.create_experiment().append_tags.assert_called_once_with('one tag')
    neptune.create_experiment().append_tags.reset_mock()

    logger.append_tags(['two', 'tags'])
    neptune.create_experiment().append_tags.assert_called_once_with('two', 'tags')


def test_neptune_leave_open_experiment_after_fit(tmpdir):
    """Verify that neptune experiment was closed after training"""
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    def _run_training(logger):
        logger._experiment = MagicMock()

        trainer_options = dict(
            default_root_dir=tmpdir,
            max_epochs=1,
            train_percent_check=0.05,
            logger=logger
        )
        trainer = Trainer(**trainer_options)
        trainer.fit(model)
        return logger

    logger_close_after_fit = _run_training(NeptuneLogger(offline_mode=True))
    assert logger_close_after_fit._experiment.stop.call_count == 1

    logger_open_after_fit = _run_training(NeptuneLogger(offline_mode=True, close_after_fit=False))
    assert logger_open_after_fit._experiment.stop.call_count == 0
