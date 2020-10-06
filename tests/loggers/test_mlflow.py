import importlib.util
import os

from unittest import mock
from unittest.mock import MagicMock

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from tests.base import EvalModelTemplate


def test_mlflow_logger_exists(tmpdir):
    """ Test launching two independent loggers. """
    logger = MLFlowLogger('test', save_dir=tmpdir)
    # same name leads to same experiment id, but different runs get recorded
    logger2 = MLFlowLogger('test', save_dir=tmpdir)
    assert logger.experiment_id == logger2.experiment_id
    assert logger.run_id != logger2.run_id
    logger3 = MLFlowLogger('new', save_dir=tmpdir)
    assert logger3.experiment_id != logger.experiment_id


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

    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        logger=logger,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=3,
    )
    trainer.fit(model)
    assert trainer.checkpoint_callback.dirpath == (tmpdir / "exp-id" / "run-id" / 'checkpoints')
    assert set(os.listdir(trainer.checkpoint_callback.dirpath)) == {'epoch=0.ckpt'}


def test_mlflow_logger_dirs_creation(tmpdir):
    """ Test that the logger creates the folders and files in the right place. """
    if not importlib.util.find_spec('mlflow'):
        pytest.xfail(f"test for explicit file creation requires mlflow dependency to be installed.")

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

    model = EvalModelTemplate()
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=1, limit_val_batches=3)
    trainer.fit(model)
    assert set(os.listdir(tmpdir / exp_id)) == {run_id, 'meta.yaml'}
    assert 'epoch' in os.listdir(tmpdir / exp_id / run_id / 'metrics')
    assert set(os.listdir(tmpdir / exp_id / run_id / 'params')) == model.hparams.keys()
    assert trainer.checkpoint_callback.dirpath == (tmpdir / exp_id / run_id / 'checkpoints')
    assert set(os.listdir(trainer.checkpoint_callback.dirpath)) == {'epoch=0.ckpt'}


def test_mlflow_experiment_id_retrieved_once(tmpdir):
    logger = MLFlowLogger('test', save_dir=tmpdir)
    get_experiment_name = logger._mlflow_client.get_experiment_by_name
    with mock.patch.object(MlflowClient, 'get_experiment_by_name', wraps=get_experiment_name) as mocked:
        _ = logger.experiment
        _ = logger.experiment
        _ = logger.experiment
        assert mocked.call_count == 1
