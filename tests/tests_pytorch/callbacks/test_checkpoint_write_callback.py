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
        """Test hook should trigger after checkpoint is written."""
        self.called = True
        self.filepath = str(filepath)
        self.file_existed = os.path.exists(filepath)

        try:
            checkpoint = torch.load(filepath, map_location="cpu")
            self.checkpoint_valid = "state_dict" in checkpoint
        except Exception:
            self.checkpoint_valid = False


def test_on_checkpoint_write_end_called(tmp_path):
    """Test that on_checkpoint_write_end is triggered after checkpoint saving."""
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


def test_on_checkpoint_write_end_exception_safe(tmp_path):
    """Test that callback exceptions don’t block others."""
    model = BoringModel()

    class FailingCallback(Callback):
        def on_checkpoint_write_end(self, trainer, pl_module, filepath):
            raise RuntimeError("Intentional error")

    class SuccessCallback(Callback):
        def __init__(self):
            self.called = False

        def on_checkpoint_write_end(self, trainer, pl_module, filepath):
            self.called = True

    fail_cb = FailingCallback()
    success_cb = SuccessCallback()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, callbacks=[fail_cb, success_cb], logger=False)

    trainer.fit(model)
    checkpoint_path = tmp_path / "test_checkpoint.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()
    assert success_cb.called


def test_checkpoint_file_accessibility(tmp_path):
    """Test that checkpoint is readable during callback execution."""
    model = BoringModel()

    class FileAccessCallback(Callback):
        def __init__(self):
            self.can_read = False
            self.valid = False

        def on_checkpoint_write_end(self, trainer, pl_module, filepath):
            try:
                ckpt = torch.load(filepath, map_location="cpu")
                self.can_read = True
                self.valid = "state_dict" in ckpt
            except (OSError, RuntimeError):
                pass

    callback = FileAccessCallback()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, callbacks=[callback], logger=False)

    trainer.fit(model)
    checkpoint_path = tmp_path / "test_checkpoint.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    assert checkpoint_path.exists()
    assert callback.can_read
    assert callback.valid
