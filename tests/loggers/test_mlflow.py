import os

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


def test_mlflow_logger_dirs_creation(tmpdir):
    """ Test that the logger creates the folders in the right place. """
    assert not os.listdir(tmpdir)
    logger = MLFlowLogger('test', save_dir=tmpdir)
    assert set(os.listdir(tmpdir)) == {'.trash'}
    run_id = logger.run_id
    exp_id = logger.experiment_id
    for i in range(2):
        _ = logger.experiment
        assert set(os.listdir(tmpdir)) == {'.trash', exp_id}
        assert set(os.listdir(tmpdir / exp_id)) == {run_id, 'meta.yaml'}

    model = EvalModelTemplate()
    trainer = Trainer(logger=logger, max_steps=3, limit_val_batches=3)
    trainer.fit(model)
    assert set(os.listdir(tmpdir / exp_id)) == {run_id, 'meta.yaml'}
    assert 'epoch' in os.listdir(tmpdir / exp_id / run_id / 'metrics')
    assert set(os.listdir(tmpdir / exp_id / run_id / 'params')) == model.hparams.keys()
