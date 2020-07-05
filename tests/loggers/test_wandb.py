import os
import pickle
import platform
from unittest import mock

import pytest
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from tests.base import EvalModelTemplate

# fake api key and user
os.environ.update(WANDB_API_KEY=('X' * 40), WANDB_ENTITY='username')


@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_wandb_logger(wandb):
    """Verify that basic functionality of wandb logger works.
    Wandb doesn't work well with pytest so we have to mock it out here."""
    logger = WandbLogger(anonymous=True, offline=True)

    logger.log_metrics({'acc': 1.0})
    wandb.init().log.assert_called_once_with({'acc': 1.0})

    wandb.init().log.reset_mock()
    logger.log_metrics({'acc': 1.0}, step=3)
    wandb.init().log.assert_called_once_with({'global_step': 3, 'acc': 1.0})

    logger.log_hyperparams({'test': None})
    wandb.init().config.update.assert_called_once_with({'test': None}, allow_val_change=True)

    logger.watch('model', 'log', 10)
    wandb.init().watch.assert_called_once_with('model', log='log', log_freq=10)

    assert logger.name == wandb.init().project_name()
    assert logger.version == wandb.init().id


@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_wandb_pickle(wandb, tmpdir):
    """Verify that pickling trainer with wandb logger works.

    Wandb doesn't work well with pytest so we have to mock it out here.
    """
    class Experiment:
        id = 'the_id'

        def project_name(self):
            return 'the_project_name'

    wandb.init.return_value = Experiment()

    logger = WandbLogger(id='the_id', offline=True)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
    )
    # Access the experiment to ensure it's created
    assert trainer.logger.experiment, 'missing experiment'
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)

    assert os.environ['WANDB_MODE'] == 'dryrun'
    assert trainer2.logger.__class__.__name__ == WandbLogger.__name__
    assert trainer2.logger.experiment, 'missing experiment'

    wandb.init.assert_called()
    assert 'id' in wandb.init.call_args[1]
    assert wandb.init.call_args[1]['id'] == 'the_id'

    del os.environ['WANDB_MODE']


@pytest.mark.skipif(
    platform.system() == 'Windows',
    reason='Cannot run in offline mode on windows without api key.'
    # known issue: https://github.com/wandb/client/issues/366
    # TODO: remove skipping when issue gets fixed
)
def test_wandb_logger_dirs_creation(tmpdir):
    """ Test that the logger creates the folders and files in the right place. """
    logger = WandbLogger(project='project', name='name', save_dir=str(tmpdir), offline=True, anonymous=True)
    assert logger.version is None
    assert logger.name is None
    assert str(tmpdir) == logger.save_dir
    assert not os.listdir(tmpdir)
    # logger.log_metrics({'x': 1})
    version = logger.version

    # multiple experiment calls should not lead to new experiment folders
    for _ in range(2):
        _ = logger.experiment

    wandb_dir = tmpdir / wandb.wandb_dir()
    runs_folders = [p for p in os.listdir(wandb_dir) if p.endswith(version)]
    assert len(runs_folders) == 1
    assert len(os.listdir(wandb_dir / runs_folders[0]))

    model = EvalModelTemplate()
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=1, limit_val_batches=3)
    trainer.fit(model)

    assert trainer.ckpt_path == trainer.weights_save_path == str(tmpdir / 'project' / version / 'checkpoints')
    assert set(os.listdir(trainer.ckpt_path)) == {'epoch=0.ckpt'}
