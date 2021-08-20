# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import CheckpointIO, DeepSpeedPlugin, SingleDevicePlugin, TPUSpawnPlugin
from pytorch_lightning.utilities.debugging_examples import BoringModel
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import _PATH
from tests.helpers.runif import RunIf


class CustomCheckpointIO(CheckpointIO):
    save_checkpoint_called: bool = False
    load_checkpoint_file_called: bool = False

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        self.save_checkpoint_called = True
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: _PATH, storage_options: Optional[Any] = None) -> Dict[str, Any]:
        self.load_checkpoint_file_called = True
        return torch.load(path)


def test_checkpoint_plugin_called(tmpdir):
    """
    Ensure that the custom checkpoint IO plugin and torch checkpoint IO plugin is called when saving/loading.
    """
    checkpoint_plugin = CustomCheckpointIO()
    checkpoint_plugin = MagicMock(wraps=checkpoint_plugin, spec=CustomCheckpointIO)

    ck = ModelCheckpoint(dirpath=tmpdir, save_last=True)

    model = BoringModel()
    device = torch.device("cpu")
    trainer = Trainer(
        default_root_dir=tmpdir,
        plugins=SingleDevicePlugin(device, checkpoint_io=checkpoint_plugin),
        callbacks=ck,
        max_epochs=1,
    )
    trainer.fit(model)
    assert checkpoint_plugin.save_checkpoint.call_count == 3
    trainer.test(model, ckpt_path=ck.last_model_path)
    checkpoint_plugin.load_checkpoint.assert_called_with(tmpdir / "last.ckpt")

    checkpoint_plugin.reset_mock()
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
    assert checkpoint_plugin.save_checkpoint.call_count == 3

    trainer.test(model, ckpt_path=ck.last_model_path)
    checkpoint_plugin.load_checkpoint.assert_called_once()
    checkpoint_plugin.load_checkpoint.assert_called_with(tmpdir / "last.ckpt")


@pytest.mark.parametrize("plugin_cls", [pytest.param(DeepSpeedPlugin, marks=RunIf(deepspeed=True)), TPUSpawnPlugin])
def test_no_checkpoint_io_plugin_support(plugin_cls):
    with pytest.raises(MisconfigurationException, match="currently does not support custom checkpoint plugins"):
        plugin_cls().checkpoint_io = CustomCheckpointIO()
