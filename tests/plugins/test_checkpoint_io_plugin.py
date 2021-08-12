from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import (
    CheckpointIOPlugin,
    DeepSpeedPlugin,
    SingleDevicePlugin,
    TorchCheckpointIOPlugin,
    TPUSpawnPlugin,
)
from pytorch_lightning.plugins.checkpoint.checkpoint import TLoadStorageOptions, TSaveStorageOptions
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class CustomCheckpointPlugin(CheckpointIOPlugin):
    save_checkpoint_called: bool = False
    load_checkpoint_file_called: bool = False

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[TSaveStorageOptions] = None
    ) -> None:
        self.save_checkpoint_called = True
        torch.save(checkpoint, path)

    def load_checkpoint(
        self, path: Union[str, Path], storage_options: Optional[TLoadStorageOptions] = None
    ) -> Dict[str, Any]:
        self.load_checkpoint_file_called = True
        return torch.load(path)


class CustomTorchCheckpointIOPlugin(TorchCheckpointIOPlugin):
    save_checkpoint_called: bool = False
    load_checkpoint_file_called: bool = False

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        self.save_checkpoint_called = True
        super().save_checkpoint(checkpoint, path)

    def load_checkpoint(
        self, path: Union[str, Path], map_location: Optional[Callable] = lambda storage, loc: storage
    ) -> Dict[str, Any]:
        self.load_checkpoint_file_called = True
        return super().load_checkpoint(path)


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

    checkpoint_plugin.save_checkpoint_called = False
    checkpoint_plugin.load_checkpoint_file_called = False
    ck = ModelCheckpoint(dirpath=tmpdir, save_last=True)

    model = BoringModel()
    device = torch.device("cpu")
    trainer = Trainer(
        default_root_dir=tmpdir,
        plugins=[SingleDevicePlugin(device), checkpoint_plugin],
        callbacks=ck,
        max_epochs=1,
    )
    trainer.fit(model)
    assert checkpoint_plugin.save_checkpoint_called
    trainer.test(model, ckpt_path=ck.last_model_path)
    assert checkpoint_plugin.load_checkpoint_file_called


@pytest.mark.parametrize("plugin_cls", [pytest.param(DeepSpeedPlugin, marks=RunIf(deepspeed=True)), TPUSpawnPlugin])
def test_no_checkpoint_io_plugin_support(plugin_cls):
    with pytest.raises(MisconfigurationException, match="currently does not support custom checkpoint plugins"):
        plugin_cls().checkpoint_plugin = CustomTorchCheckpointIOPlugin()
