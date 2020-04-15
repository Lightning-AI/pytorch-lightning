import os
import pickle
from unittest.mock import patch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


@patch('pytorch_lightning.loggers.wandb.wandb')
def test_wandb_logger(wandb):
    """Verify that basic functionality of wandb logger works.
    Wandb doesn't work well with pytest so we have to mock it out here."""
    tutils.reset_seed()

    logger = WandbLogger(anonymous=True, offline=True)

    logger.log_metrics({'acc': 1.0})
    wandb.init().log.assert_called_once_with({'acc': 1.0})

    wandb.init().log.reset_mock()
    logger.log_metrics({'acc': 1.0}, step=3)
    wandb.init().log.assert_called_once_with({'global_step': 3, 'acc': 1.0})

    logger.log_hyperparams({'test': None})
    wandb.init().config.update.assert_called_once_with({'test': None})

    logger.watch('model', 'log', 10)
    wandb.init().watch.assert_called_once_with('model', log='log', log_freq=10)

    assert logger.name == wandb.init().project_name()
    assert logger.version == wandb.init().id


@patch('pytorch_lightning.loggers.wandb.wandb')
def test_wandb_pickle(wandb):
    """Verify that pickling trainer with wandb logger works.

    Wandb doesn't work well with pytest so we have to mock it out here.
    """
    tutils.reset_seed()

    class Experiment:
        id = 'the_id'

    wandb.init.return_value = Experiment()

    logger = WandbLogger(id='the_id', offline=True)

    trainer_options = dict(max_epochs=1, logger=logger)

    trainer = Trainer(**trainer_options)
    # Access the experiment to ensure it's created
    trainer.logger.experiment
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)

    assert os.environ['WANDB_MODE'] == 'dryrun'
    assert trainer2.logger.__class__.__name__ == WandbLogger.__name__
    _ = trainer2.logger.experiment

    wandb.init.assert_called()
    assert 'id' in wandb.init.call_args[1]
    assert wandb.init.call_args[1]['id'] == 'the_id'

    del os.environ['WANDB_MODE']
