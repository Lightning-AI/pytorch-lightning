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
from unittest import mock
from unittest.mock import MagicMock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import _MLFLOW_AVAILABLE, MLFlowLogger
from tests.helpers import BoringModel


def mock_mlflow_run_creation(logger, experiment_name=None, experiment_id=None, run_id=None):
    """ Helper function to simulate mlflow client creating a new (or existing) experiment. """
    run = MagicMock()
    run.info.run_id = run_id
    logger._mlflow_client.get_experiment_by_name = MagicMock(return_value=experiment_name)
    logger._mlflow_client.create_experiment = MagicMock(return_value=experiment_id)
    logger._mlflow_client.create_run = MagicMock(return_value=run)
    return logger


@mock.patch('pytorch_lightning.loggers.mlflow.mlflow')
@mock.patch('pytorch_lightning.loggers.mlflow.MlflowClient')
def test_mlflow_logger_exists(client, mlflow, tmpdir):
    """ Test launching three independent loggers with either same or different experiment name. """

    run1 = MagicMock()
    run1.info.run_id = "run-id-1"

    run2 = MagicMock()
    run2.info.run_id = "run-id-2"

    run3 = MagicMock()
    run3.info.run_id = "run-id-3"

    # simulate non-existing experiment creation
    client.return_value.get_experiment_by_name = MagicMock(return_value=None)
    client.return_value.create_experiment = MagicMock(return_value="exp-id-1")  # experiment_id
    client.return_value.create_run = MagicMock(return_value=run1)

    logger = MLFlowLogger('test', save_dir=tmpdir)
    assert logger._experiment_id is None
    assert logger._run_id is None
    _ = logger.experiment
    assert logger.experiment_id == "exp-id-1"
    assert logger.run_id == "run-id-1"
    assert logger.experiment.create_experiment.asset_called_once()
    client.reset_mock(return_value=True)

    # simulate existing experiment returns experiment id
    exp1 = MagicMock()
    exp1.experiment_id = "exp-id-1"
    client.return_value.get_experiment_by_name = MagicMock(return_value=exp1)
    client.return_value.create_run = MagicMock(return_value=run2)

    # same name leads to same experiment id, but different runs get recorded
    logger2 = MLFlowLogger('test', save_dir=tmpdir)
    assert logger2.experiment_id == logger.experiment_id
    assert logger2.run_id == "run-id-2"
    assert logger2.experiment.create_experiment.call_count == 0
    assert logger2.experiment.create_run.asset_called_once()
    client.reset_mock(return_value=True)

    # simulate a 3rd experiment with new name
    client.return_value.get_experiment_by_name = MagicMock(return_value=None)
    client.return_value.create_experiment = MagicMock(return_value="exp-id-3")
    client.return_value.create_run = MagicMock(return_value=run3)

    # logger with new experiment name causes new experiment id and new run id to be created
    logger3 = MLFlowLogger('new', save_dir=tmpdir)
    assert logger3.experiment_id == "exp-id-3" != logger.experiment_id
    assert logger3.run_id == "run-id-3"


@mock.patch("pytorch_lightning.loggers.mlflow.mlflow")
@mock.patch("pytorch_lightning.loggers.mlflow.MlflowClient")
def test_mlflow_log_dir(client, mlflow, tmpdir):
    """ Test that the trainer saves checkpoints in the logger's save dir. """

    # simulate experiment creation with mlflow client mock
    run = MagicMock()
    run.info.run_id = "run-id"
    client.return_value.get_experiment_by_name = MagicMock(return_value=None)
    client.return_value.create_experiment = MagicMock(return_value="exp-id")
    client.return_value.create_run = MagicMock(return_value=run)

    # test construction of default log dir path
    logger = MLFlowLogger("test", save_dir=tmpdir)
    assert logger.save_dir == tmpdir
    assert logger.version == "run-id"
    assert logger.name == "exp-id"

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        logger=logger,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=3,
    )
    assert trainer.log_dir == logger.save_dir
    trainer.fit(model)
    assert trainer.checkpoint_callback.dirpath == (tmpdir / "exp-id" / "run-id" / 'checkpoints')
    assert set(os.listdir(trainer.checkpoint_callback.dirpath)) == {'epoch=0-step=0.ckpt'}
    assert trainer.log_dir == logger.save_dir


def test_mlflow_logger_dirs_creation(tmpdir):
    """ Test that the logger creates the folders and files in the right place. """
    if not _MLFLOW_AVAILABLE:
        pytest.xfail("test for explicit file creation requires mlflow dependency to be installed.")

    assert not os.listdir(tmpdir)
    logger = MLFlowLogger('test', save_dir=tmpdir)
    assert logger.save_dir == tmpdir
    assert set(os.listdir(tmpdir)) == {'.trash'}
    run_id = logger.run_id
    exp_id = logger.experiment_id

    # multiple experiment calls should not lead to new experiment folders
    for i in range(2):
        _ = logger.experiment
        assert set(os.listdir(tmpdir)) == {'.trash', exp_id}
        assert set(os.listdir(tmpdir / exp_id)) == {run_id, 'meta.yaml'}

    class CustomModel(BoringModel):

        def training_epoch_end(self, *args, **kwargs):
            super().training_epoch_end(*args, **kwargs)
            self.log('epoch', self.current_epoch)

    model = CustomModel()
    limit_batches = 5
    trainer = Trainer(
        default_root_dir=tmpdir,
        logger=logger,
        max_epochs=1,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        log_gpu_memory=True,
    )
    trainer.fit(model)
    assert set(os.listdir(tmpdir / exp_id)) == {run_id, 'meta.yaml'}
    assert 'epoch' in os.listdir(tmpdir / exp_id / run_id / 'metrics')
    assert set(os.listdir(tmpdir / exp_id / run_id / 'params')) == model.hparams.keys()
    assert trainer.checkpoint_callback.dirpath == (tmpdir / exp_id / run_id / 'checkpoints')
    assert os.listdir(trainer.checkpoint_callback.dirpath) == [f'epoch=0-step={limit_batches - 1}.ckpt']


@mock.patch('pytorch_lightning.loggers.mlflow.mlflow')
@mock.patch('pytorch_lightning.loggers.mlflow.MlflowClient')
def test_mlflow_experiment_id_retrieved_once(client, mlflow, tmpdir):
    """
    Test that the logger experiment_id retrieved only once.
    """
    logger = MLFlowLogger('test', save_dir=tmpdir)
    _ = logger.experiment
    _ = logger.experiment
    _ = logger.experiment
    assert logger.experiment.get_experiment_by_name.call_count == 1


@mock.patch('pytorch_lightning.loggers.mlflow.mlflow')
@mock.patch('pytorch_lightning.loggers.mlflow.MlflowClient')
def test_mlflow_logger_with_unexpected_characters(client, mlflow, tmpdir):
    """
    Test that the logger raises warning with special characters not accepted by MLFlow.
    """
    logger = MLFlowLogger('test', save_dir=tmpdir)
    metrics = {'[some_metric]': 10}

    with pytest.warns(RuntimeWarning, match='special characters in metric name'):
        logger.log_metrics(metrics)


@mock.patch('pytorch_lightning.loggers.mlflow.mlflow')
@mock.patch('pytorch_lightning.loggers.mlflow.MlflowClient')
def test_mlflow_logger_with_long_param_value(client, mlflow, tmpdir):
    """
    Test that the logger raises warning with special characters not accepted by MLFlow.
    """
    logger = MLFlowLogger('test', save_dir=tmpdir)
    value = 'test' * 100
    key = 'test_param'
    params = {key: value}

    with pytest.warns(RuntimeWarning, match=f'Discard {key}={value}'):
        logger.log_hyperparams(params)
