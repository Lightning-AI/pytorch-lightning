from pathlib import Path
from typing import Any, Dict, Union

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import CheckpointIOPlugin, SingleDevicePlugin, TorchCheckpointIOPlugin
from tests.helpers.boring_model import BoringModel


class CustomCheckpointPlugin(CheckpointIOPlugin):
    save_checkpoint_called: bool = False
    load_checkpoint_file_called: bool = False

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: Union[str, Path]) -> None:
        self.save_checkpoint_called = True
        torch.save(checkpoint, path)

    def load_checkpoint_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        self.load_checkpoint_file_called = True
        return torch.load(path)


class CustomTorchCheckpointIOPlugin(TorchCheckpointIOPlugin):
    save_checkpoint_called: bool = False
    load_checkpoint_file_called: bool = False

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: Union[str, Path]) -> None:
        self.save_checkpoint_called = True
        super().save_checkpoint(checkpoint, path)

    def load_checkpoint_file(self, path: Union[str, Path]) -> Dict[str, Any]:
        self.load_checkpoint_file_called = True
        return super().load_checkpoint_file(path)


@pytest.mark.parametrize("checkpoint_plugin", [CustomTorchCheckpointIOPlugin(), CustomCheckpointPlugin()])
def test_checkpoint_plugin_called(tmpdir, checkpoint_plugin):
    """
    Ensure that the custom checkpoint IO plugin and torch checkpoint IO plugin is called when saving/loading.
    """

    ck = ModelCheckpoint(dirpath=tmpdir, save_last=True)

    model = BoringModel()
    device = torch.device("cpu")
    trainer = Trainer(
        default_root_dir=tmpdir,
        plugins=SingleDevicePlugin(device, checkpoint_plugin=checkpoint_plugin),
        callbacks=ck,
        max_epochs=1,
    )
    trainer.fit(model)
    assert checkpoint_plugin.save_checkpoint_called
    trainer.test(model, ckpt_path=ck.last_model_path)
    assert checkpoint_plugin.load_checkpoint_file_called
