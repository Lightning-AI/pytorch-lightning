# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pickle
import types
from argparse import ArgumentParser
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel, EvalModelTemplate


def get_warnings(recwarn):
    warnings_text = '\n'.join(str(w.message) for w in recwarn.list)
    recwarn.clear()
    return warnings_text


@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_wandb_logger_init(wandb, recwarn):
    """Verify that basic functionality of wandb logger works.
    Wandb doesn't work well with pytest so we have to mock it out here."""

    # test wandb.init called when there is no W&B run
    wandb.run = None
    logger = WandbLogger()
    logger.log_metrics({'acc': 1.0})
    wandb.init.assert_called_once()
    wandb.init().log.assert_called_once_with({'acc': 1.0}, step=None)

    # mock wandb step
    wandb.init().step = 0

    # test wandb.init not called if there is a W&B run
    wandb.init().log.reset_mock()
    wandb.init.reset_mock()
    wandb.run = wandb.init()
    logger = WandbLogger()
    logger.log_metrics({'acc': 1.0}, step=3)
    wandb.init.assert_called_once()
    wandb.init().log.assert_called_once_with({'acc': 1.0}, step=3)

    # continue training on same W&B run and offset step
    wandb.init().step = 3
    logger.finalize('success')
    logger.log_metrics({'acc': 1.0}, step=3)
    wandb.init().log.assert_called_with({'acc': 1.0}, step=6)

    # log hyper parameters
    logger.log_hyperparams({'test': None, 'nested': {'a': 1}, 'b': [2, 3, 4]})
    wandb.init().config.update.assert_called_once_with(
        {'test': 'None', 'nested/a': 1, 'b': [2, 3, 4]},
        allow_val_change=True,
    )

    # watch a model
    logger.watch('model', 'log', 10)
    wandb.init().watch.assert_called_once_with('model', log='log', log_freq=10)

    # verify warning for logging at a previous step
    assert 'Trying to log at a previous step' not in get_warnings(recwarn)
    # current step from wandb should be 6 (last logged step)
    logger.experiment.step = 6
    # logging at step 2 should raise a warning (step_offset is still 3)
    logger.log_metrics({'acc': 1.0}, step=2)
    assert 'Trying to log at a previous step' in get_warnings(recwarn)
    # logging again at step 2 should not display again the same warning
    logger.log_metrics({'acc': 1.0}, step=2)
    assert 'Trying to log at a previous step' not in get_warnings(recwarn)

    assert logger.name == wandb.init().project_name()
    assert logger.version == wandb.init().id


@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_wandb_pickle(wandb, tmpdir):
    """
    Verify that pickling trainer with wandb logger works.
    Wandb doesn't work well with pytest so we have to mock it out here.
    """
    class Experiment:
        """ """
        id = 'the_id'
        step = 0
        dir = 'wandb'

        def project_name(self):
            return 'the_project_name'

    wandb.run = None
    wandb.init.return_value = Experiment()
    logger = WandbLogger(id='the_id', offline=True)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
    )
    # Access the experiment to ensure it's created
    assert trainer.logger.experiment, 'missing experiment'
    assert trainer.log_dir == logger.save_dir
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)

    assert os.environ['WANDB_MODE'] == 'dryrun'
    assert trainer2.logger.__class__.__name__ == WandbLogger.__name__
    assert trainer2.logger.experiment, 'missing experiment'

    wandb.init.assert_called()
    assert 'id' in wandb.init.call_args[1]
    assert wandb.init.call_args[1]['id'] == 'the_id'

    del os.environ['WANDB_MODE']


@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_wandb_logger_dirs_creation(wandb, tmpdir):
    """ Test that the logger creates the folders and files in the right place. """
    logger = WandbLogger(save_dir=str(tmpdir), offline=True)
    assert logger.version is None
    assert logger.name is None

    # mock return values of experiment
    wandb.run = None
    wandb.init().step = 0
    logger.experiment.id = '1'
    logger.experiment.project_name.return_value = 'project'
    logger.experiment.step = 0

    for _ in range(2):
        _ = logger.experiment

    assert logger.version == '1'
    assert logger.name == 'project'
    assert str(tmpdir) == logger.save_dir
    assert not os.listdir(tmpdir)

    version = logger.version
    model = EvalModelTemplate()
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=1, limit_val_batches=3, log_every_n_steps=1)
    assert trainer.log_dir == logger.save_dir
    trainer.fit(model)

    assert trainer.checkpoint_callback.dirpath == str(tmpdir / 'project' / version / 'checkpoints')
    assert set(os.listdir(trainer.checkpoint_callback.dirpath)) == {'epoch=0-step=9.ckpt'}
    assert trainer.log_dir == logger.save_dir


def test_wandb_sanitize_callable_params(tmpdir):
    """
    Callback function are not serializiable. Therefore, we get them a chance to return
    something and if the returned type is not accepted, return None.
    """
    opt = "--max_epochs 1".split(" ")
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parent_parser=parser)
    params = parser.parse_args(opt)

    def return_something():
        return "something"
    params.something = return_something

    def wrapper_something():
        return return_something

    params.wrapper_something_wo_name = lambda: lambda: '1'
    params.wrapper_something = wrapper_something

    assert isinstance(params.gpus, types.FunctionType)
    params = WandbLogger._convert_params(params)
    params = WandbLogger._flatten_dict(params)
    params = WandbLogger._sanitize_callable_params(params)
    assert params["gpus"] == '_gpus_arg_default'
    assert params["something"] == "something"
    assert params["wrapper_something"] == "wrapper_something"
    assert params["wrapper_something_wo_name"] == "<lambda>"


@mock.patch('pytorch_lightning.loggers.wandb.wandb')
def test_wandb_logger_offline_log_model(wandb, tmpdir):
    """ Test that log_model=True raises an error in offline mode """
    with pytest.raises(MisconfigurationException, match='checkpoints cannot be uploaded in offline mode'):
        logger = WandbLogger(save_dir=str(tmpdir), offline=True, log_model=True)
