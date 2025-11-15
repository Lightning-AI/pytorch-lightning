import os

import torch

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


class CheckpointWriteEndCallback(Callback):
    def __init__(self):
        self.called = False
        self.filepath = None
        self.file_existed = False
        self.checkpoint_valid = False

    def on_checkpoint_write_end(self, trainer, pl_module, filepath):
        """Verify that the hook triggers after checkpoint is written."""
        self.called = True
        self.filepath = str(filepath)
        self.file_existed = os.path.exists(filepath)

        checkpoint = torch.load(filepath, map_location="cpu")
        self.checkpoint_valid = "state_dict" in checkpoint


def test_on_checkpoint_write_end_called(tmp_path):
    """Test that on_checkpoint_write_end is called after saving a checkpoint."""
    model = BoringModel()
    callback = CheckpointWriteEndCallback()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, callbacks=[callback], logger=False)

    trainer.fit(model)

    checkpoint_path = tmp_path / "test_checkpoint.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()
    assert callback.called
    assert callback.file_existed
    assert callback.checkpoint_valid
    assert callback.filepath == str(checkpoint_path)
