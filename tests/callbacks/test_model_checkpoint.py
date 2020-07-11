import os
import pickle
import platform
from pathlib import Path

import cloudpickle
import pytest

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tests.base import EvalModelTemplate


@pytest.mark.parametrize('save_top_k', [-1, 0, 1, 2])
def test_model_checkpoint_with_non_string_input(tmpdir, save_top_k):
    """ Test that None in checkpoint callback is valid and that chkp_path is set correctly """
    tutils.reset_seed()
    model = EvalModelTemplate()

    checkpoint = ModelCheckpoint(filepath=None, save_top_k=save_top_k)

    trainer = Trainer(default_root_dir=tmpdir, checkpoint_callback=checkpoint, overfit_batches=0.20, max_epochs=2)
    trainer.fit(model)
    assert checkpoint.dirpath == tmpdir / trainer.logger.name / 'version_0' / 'checkpoints'


@pytest.mark.parametrize(
    'logger_version,expected', [(None, 'version_0'), (1, 'version_1'), ('awesome', 'awesome')],
)
def test_model_checkpoint_path(tmpdir, logger_version, expected):
    """Test that "version_" prefix is only added when logger's version is an integer"""
    tutils.reset_seed()
    model = EvalModelTemplate()
    logger = TensorBoardLogger(str(tmpdir), version=logger_version)

    trainer = Trainer(default_root_dir=tmpdir, overfit_batches=0.2, max_epochs=2, logger=logger)
    trainer.fit(model)

    ckpt_version = Path(trainer.checkpoint_callback.dirpath).parent.name
    assert ckpt_version == expected


@pytest.mark.parametrize(
    'filepath,filename,tgt_dir,tgt_filename',
    [
        (None, None, 'lightning_logs/version_0/checkpoints', 'epoch=4.ckpt'),
        (None, 'test_{epoch}', 'lightning_logs/version_0/checkpoints', 'test_epoch=4.ckpt'),
        ('checkpoints', None, 'checkpoints', 'epoch=4.ckpt'),
        ('checkpoints', '{v_num}', 'checkpoints', 'v_num=0_v0.ckpt'),
        ('checkpoints/{v_num}', None, 'checkpoints', 'v_num=0.ckpt'),
        ('checkpoints/{v_num}', None, 'checkpoints', 'v_num=0_v0.ckpt')
    ],
)
def test_model_checkpoint_filename(tmpdir, filepath, filename, tgt_dir, tgt_filename):
    """Test that the checkpoint path is built from filepath and filename"""
    tutils.reset_seed()
    model = EvalModelTemplate()

    if filepath is not None:
        filepath = tmpdir / filepath
        os.makedirs(tmpdir / tgt_dir)

    checkpoint = ModelCheckpoint(filepath=filepath, filename=filename, save_top_k=-1)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        checkpoint_callback=checkpoint
    )
    trainer.fit(model)

    if filename is None:
        filename = '{epoch}'

    assert os.path.relpath(trainer.ckpt_path, tmpdir) == tgt_dir
    assert os.path.relpath(checkpoint.dirpath, tmpdir) == tgt_dir
    assert os.path.exists(os.path.join(trainer.ckpt_path, tgt_filename))


def test_pickling(tmpdir):
    ckpt = ModelCheckpoint(tmpdir)

    ckpt_pickled = pickle.dumps(ckpt)
    ckpt_loaded = pickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)

    ckpt_pickled = cloudpickle.dumps(ckpt)
    ckpt_loaded = cloudpickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)


class ModelCheckpointTestInvocations(ModelCheckpoint):
    # this class has to be defined outside the test function, otherwise we get pickle error
    # due to the way ddp process is launched

    def __init__(self, expected_count, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.expected_count = expected_count

    def _save_model(self, filepath, trainer, pl_module):
        # make sure we don't save twice
        assert not os.path.isfile(filepath)
        self.count += 1
        super()._save_model(filepath, trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        # on rank 0 we expect the saved files and on all others no saves
        assert (trainer.global_rank == 0 and self.count == self.expected_count) or (
            trainer.global_rank > 0 and self.count == 0
        )


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_model_checkpoint_no_extraneous_invocations(tmpdir):
    """Test to ensure that the model callback saves the checkpoints only once in distributed mode."""
    model = EvalModelTemplate()
    num_epochs = 4
    model_checkpoint = ModelCheckpointTestInvocations(expected_count=num_epochs, save_top_k=-1)
    trainer = Trainer(
        distributed_backend='ddp_cpu',
        num_processes=2,
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=model_checkpoint,
        max_epochs=num_epochs,
    )
    result = trainer.fit(model)
    assert 1 == result
