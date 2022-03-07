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
import os
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities.types import _PATH
from tests.helpers.boring_model import BoringModel


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: _PATH, storage_options: Optional[Any] = None) -> Dict[str, Any]:
        return torch.load(path)

    def remove_checkpoint(self, path: _PATH) -> None:
        os.remove(path)


def test_checkpoint_plugin_called(tmpdir):
    """Ensure that the custom checkpoint IO plugin and torch checkpoint IO plugin is called when saving/loading."""
    checkpoint_plugin = CustomCheckpointIO()
    checkpoint_plugin = MagicMock(wraps=checkpoint_plugin, spec=CustomCheckpointIO)

    ck = ModelCheckpoint(dirpath=tmpdir, save_last=True)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        strategy=SingleDeviceStrategy("cpu", checkpoint_io=checkpoint_plugin),
        callbacks=ck,
        max_epochs=2,
    )
    trainer.fit(model)

    assert checkpoint_plugin.save_checkpoint.call_count == 4
    assert checkpoint_plugin.remove_checkpoint.call_count == 1

    trainer.test(model, ckpt_path=ck.last_model_path)
    checkpoint_plugin.load_checkpoint.assert_called_with(tmpdir / "last.ckpt")

    checkpoint_plugin.reset_mock()
    ck = ModelCheckpoint(dirpath=tmpdir, save_last=True)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        strategy=SingleDeviceStrategy("cpu"),
        plugins=[checkpoint_plugin],
        callbacks=ck,
        max_epochs=2,
    )
    trainer.fit(model)

    assert checkpoint_plugin.save_checkpoint.call_count == 4
    assert checkpoint_plugin.remove_checkpoint.call_count == 1

    trainer.test(model, ckpt_path=ck.last_model_path)
    checkpoint_plugin.load_checkpoint.assert_called_once()
    checkpoint_plugin.load_checkpoint.assert_called_with(tmpdir / "last.ckpt")
