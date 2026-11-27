# Copyright The Lightning AI team.
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
from unittest.mock import PropertyMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, OnExceptionCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.combined_loader import CombinedLoader


@pytest.mark.parametrize("weights_only", [False, True])
def test_training_metadata_before_dataloader_setup(tmp_path, weights_only):
    trainer = Trainer(accelerator="cpu", devices=1, default_root_dir=tmp_path, logger=False, enable_checkpointing=False)
    trainer.strategy.connect(BoringModel())
    path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(path, weights_only=weights_only)
    checkpoint = torch.load(path, weights_only=True)
    if weights_only:
        assert "training_metadata" not in checkpoint
    else:
        assert checkpoint["training_metadata"] == {
            "num_devices": 1,
            "num_nodes": 1,
            "world_size": 1,
            "num_workers": [],
        }


def test_training_metadata_multiple_loaders(tmp_path):
    trainer = Trainer(accelerator="cpu", devices=1, default_root_dir=tmp_path, logger=False, enable_checkpointing=False)
    trainer.strategy.connect(BoringModel())
    # Include nested loaders and a custom iterable with no worker count. Do not start any workers.
    trainer.fit_loop._combined_loader = CombinedLoader({
        "first": DataLoader([1], num_workers=2),
        "nested": [DataLoader([1], num_workers=0), range(2)],
    })
    with (
        patch.object(Trainer, "num_devices", new_callable=PropertyMock, return_value=4),
        patch.object(Trainer, "num_nodes", new_callable=PropertyMock, return_value=2),
        patch.object(Trainer, "world_size", new_callable=PropertyMock, return_value=8),
    ):
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
    assert checkpoint["training_metadata"] == {
        "num_devices": 4,
        "num_nodes": 2,
        "world_size": 8,
        "num_workers": [2, 0, None],
    }


@pytest.mark.parametrize("callback_type", [ModelCheckpoint, OnExceptionCheckpoint])
def test_training_metadata_on_exception(tmp_path, callback_type):
    class RaiseAfterBatch(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            raise RuntimeError("Interrupted training")

    kwargs = {"save_on_exception": True} if callback_type is ModelCheckpoint else {}
    callback = callback_type(dirpath=tmp_path, filename="exception", **kwargs)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
        callbacks=[callback, RaiseAfterBatch()],
        logger=False,
        enable_progress_bar=False,
        max_steps=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    with pytest.raises(RuntimeError, match="Interrupted training"):
        trainer.fit(BoringModel())
    checkpoint = torch.load(tmp_path / "exception.ckpt", weights_only=True)
    assert checkpoint["training_metadata"] == {
        "num_devices": 1,
        "num_nodes": 1,
        "world_size": 1,
        "num_workers": [0],
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_training_metadata_resume(tmp_path, legacy):
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        max_steps=1,
        limit_val_batches=0,
    )
    trainer.fit(BoringModel())
    path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(path, weights_only=False)
    checkpoint = torch.load(path, weights_only=True)
    if legacy:
        del checkpoint["training_metadata"]
    else:
        # Metadata is informational: resuming with a different topology remains supported.
        checkpoint["training_metadata"]["world_size"] = 8
    torch.save(checkpoint, path)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        max_steps=2,
        limit_val_batches=0,
    )
    trainer.fit(BoringModel(), ckpt_path=path)
    assert trainer.global_step == 2
