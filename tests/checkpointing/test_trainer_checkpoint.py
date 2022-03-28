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
from unittest import mock
from unittest.mock import ANY

import pytest
import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO
from pytorch_lightning.plugins.io.xla_plugin import XLACheckpointIO
from tests.helpers import BoringModel


def test_finetuning_with_ckpt_path(tmpdir):
    """This test validates that generated ModelCheckpoint is pointing to the right best_model_path during test."""

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=tmpdir, filename="{epoch:02d}", save_top_k=-1)

    class ExtendedBoringModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.001)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    model = ExtendedBoringModel()
    model.validation_epoch_end = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=12,
        limit_val_batches=6,
        limit_test_batches=12,
        callbacks=[checkpoint_callback],
        logger=False,
    )
    trainer.fit(model)
    assert os.listdir(tmpdir) == ["epoch=00.ckpt"]

    best_model_paths = [checkpoint_callback.best_model_path]
    for idx in range(3, 6):
        # load from checkpoint
        trainer = pl.Trainer(
            default_root_dir=tmpdir,
            max_epochs=idx,
            limit_train_batches=12,
            limit_val_batches=12,
            limit_test_batches=12,
            enable_progress_bar=False,
        )
        trainer.fit(model, ckpt_path=best_model_paths[-1])
        trainer.test()
        best_model_paths.append(trainer.checkpoint_callback.best_model_path)

    for idx, best_model_path in enumerate(best_model_paths):
        if idx == 0:
            assert best_model_path.endswith(f"epoch=0{idx}.ckpt")
        else:
            assert f"epoch={idx + 1}" in best_model_path


def test_trainer_save_checkpoint_storage_options(tmpdir):
    """This test validates that storage_options argument is properly passed to ``CheckpointIO``"""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    instance_path = tmpdir + "/path.ckpt"
    instance_storage_options = "my instance storage options"

    with mock.patch("pytorch_lightning.plugins.io.torch_plugin.TorchCheckpointIO.save_checkpoint") as io_mock:
        trainer.save_checkpoint(instance_path, storage_options=instance_storage_options)
        io_mock.assert_called_with(ANY, instance_path, storage_options=instance_storage_options)
        trainer.save_checkpoint(instance_path)
        io_mock.assert_called_with(ANY, instance_path, storage_options=None)

    with mock.patch(
        "pytorch_lightning.trainer.connectors.checkpoint_connector.CheckpointConnector.save_checkpoint"
    ) as cc_mock:
        trainer.save_checkpoint(instance_path, True)
        cc_mock.assert_called_with(instance_path, weights_only=True, storage_options=None)
        trainer.save_checkpoint(instance_path, False, instance_storage_options)
        cc_mock.assert_called_with(instance_path, weights_only=False, storage_options=instance_storage_options)

    torch_checkpoint_io = TorchCheckpointIO()
    with pytest.raises(
        TypeError,
        match=r"`Trainer.save_checkpoint\(..., storage_options=...\)` with `storage_options` arg"
        f" is not supported for `{torch_checkpoint_io.__class__.__name__}`. Please implement your custom `CheckpointIO`"
        " to define how you'd like to use `storage_options`.",
    ):
        torch_checkpoint_io.save_checkpoint({}, instance_path, storage_options=instance_storage_options)
    xla_checkpoint_io = XLACheckpointIO()
    with pytest.raises(
        TypeError,
        match=r"`Trainer.save_checkpoint\(..., storage_options=...\)` with `storage_options` arg"
        f" is not supported for `{xla_checkpoint_io.__class__.__name__}`. Please implement your custom `CheckpointIO`"
        " to define how you'd like to use `storage_options`.",
    ):
        xla_checkpoint_io.save_checkpoint({}, instance_path, storage_options=instance_storage_options)
