import os

import tests.models.utils as tutils
from pytorch_lightning.loggers import (
    WandbLogger
)


def test_wandb_logger(tmpdir):
    """Verify that basic functionality of wandb logger works."""
    tutils.reset_seed()

    wandb_dir = os.path.join(tmpdir, "wandb")
    _ = WandbLogger(save_dir=wandb_dir, anonymous=True, offline=True)


def test_wandb_pickle(tmpdir):
    """Verify that pickling trainer with wandb logger works."""
    tutils.reset_seed()

    wandb_dir = str(tmpdir)
    logger = WandbLogger(save_dir=wandb_dir, anonymous=True, offline=True)
    assert logger is not None
