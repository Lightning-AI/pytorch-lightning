import os
import pickle
import platform
from pathlib import Path

import cloudpickle
import pytest
import torch

import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.model_checkpoint import CHECKPOINT_NAME_LAST, CHECKPOINT_STATE_BEST_SCORE, \
    CHECKPOINT_STATE_BEST_PATH
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


def test_model_checkpoint_save_last_checkpoint_contents(tmpdir):
    model = EvalModelTemplate()
    num_epochs = 3
    model_checkpoint = ModelCheckpoint(filepath=tmpdir, save_top_k=num_epochs, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        early_stop_callback=False,
        checkpoint_callback=model_checkpoint,
        max_epochs=num_epochs,
    )
    trainer.fit(model)
    print(tmpdir)
    path_last_epoch = model_checkpoint.format_checkpoint_name(num_epochs - 1, {})
    path_last = tmpdir / CHECKPOINT_NAME_LAST
    ckpt_last_epoch = torch.load(str(path_last_epoch))
    ckpt_last = torch.load(str(path_last))
    assert ckpt_last[CHECKPOINT_STATE_BEST_SCORE] == ckpt_last_epoch[CHECKPOINT_STATE_BEST_SCORE]
    assert ckpt_last[CHECKPOINT_STATE_BEST_PATH] == ckpt_last_epoch[CHECKPOINT_STATE_BEST_PATH]
