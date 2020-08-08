import pytest
import os

import torch

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.upgrade_checkpoint import upgrade_checkpoint


@pytest.mark.parametrize(
    "old_checkpoint, new_checkpoint",
    [
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best": 0.34},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_score": 0.34}}},
        ),
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best_model_score": 0.99},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_score": 0.99}}},
        ),
        (
            {"epoch": 1, "global_step": 23, "checkpoint_callback_best_model_path": 'path'},
            {"epoch": 1, "global_step": 23, "callbacks": {ModelCheckpoint: {"best_model_path": 'path'}}},
        ),
        (
            {"epoch": 1, "global_step": 23, "early_stop_callback_wait": 2, "early_stop_callback_patience": 4},
            {"epoch": 1, "global_step": 23, "callbacks": {EarlyStopping: {"wait_count": 2, "patience": 4}}},
        ),
    ],
)
def test_upgrade_checkpoint(tmpdir, old_checkpoint, new_checkpoint):
    filepath = os.path.join(tmpdir, "model.ckpt")
    torch.save(old_checkpoint, filepath)
    upgrade_checkpoint(filepath)
    updated_checkpoint = torch.load(filepath)
    assert updated_checkpoint == new_checkpoint
