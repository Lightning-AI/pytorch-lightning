import pickle
from pathlib import Path

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

    trainer = Trainer(
        default_root_dir=tmpdir,
        checkpoint_callback=checkpoint,
        overfit_pct=0.20,
        max_epochs=5,
    )
    trainer.fit(model)

    # These should be different if the dirpath has be overridden
    assert trainer.ckpt_path != trainer.default_root_dir


@pytest.mark.parametrize(
    'logger_version,expected',
    [(None, 'version_0'), (1, 'version_1'), ('awesome', 'awesome')],
)
def test_model_checkpoint_path(tmpdir, logger_version, expected):
    """Test that "version_" prefix is only added when logger's version is an integer"""
    tutils.reset_seed()
    model = EvalModelTemplate()
    logger = TensorBoardLogger(str(tmpdir), version=logger_version)

    trainer = Trainer(
        default_root_dir=tmpdir,
        overfit_pct=0.2,
        max_epochs=5,
        logger=logger,
    )
    trainer.fit(model)

    ckpt_version = Path(trainer.ckpt_path).parent.name
    assert ckpt_version == expected


def test_pickling(tmpdir):
    ckpt = ModelCheckpoint(tmpdir)
    ckpt_pickled = pickle.dumps(ckpt)
    ckpt_loaded = pickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)
