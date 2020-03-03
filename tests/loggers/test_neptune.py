import pickle

from unittest.mock import patch

import tests.models.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import NeptuneLogger
from tests.models import LightningTestModel

import torch


def test_neptune_logger(tmpdir):
    """Verify that basic functionality of neptune logger works."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)
    logger = NeptuneLogger(offline_mode=True)

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, 'Training failed'


@patch('pytorch_lightning.loggers.neptune.neptune')
def test_neptune_online(neptune):
    logger = NeptuneLogger(api_key='test', project_name='project')
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


def test_neptune_pickle(tmpdir):
    """Verify that pickling trainer with neptune logger works."""
    tutils.reset_seed()

    logger = NeptuneLogger(offline_mode=True)

    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({'acc': 1.0})
