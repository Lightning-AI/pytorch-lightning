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
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load
from tests.helpers import BoringModel, RandomDataset


def test_finetuning_with_resume_from_checkpoint(tmpdir):
    """
    This test validates that generated ModelCheckpoint is pointing to the right best_model_path during test
    """

    seed_everything(4)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=tmpdir, filename="{epoch:02d}", save_top_k=-1)

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
    assert os.listdir(tmpdir) == ['epoch=00.ckpt']

    best_model_paths = [checkpoint_callback.best_model_path]
    results = []

    for idx in range(3, 6):
        # load from checkpoint
        trainer = pl.Trainer(
            default_root_dir=tmpdir,
            max_epochs=idx,
            limit_train_batches=12,
            limit_val_batches=12,
            limit_test_batches=12,
            resume_from_checkpoint=best_model_paths[-1],
            progress_bar_refresh_rate=0,
        )
        trainer.fit(model)
        trainer.test()
        results.append(deepcopy(trainer.callback_metrics))
        best_model_paths.append(trainer.checkpoint_callback.best_model_path)

    for idx in range(len(results) - 1):
        assert results[idx]["val_loss"] > results[idx + 1]["val_loss"]

    for idx, best_model_path in enumerate(best_model_paths):
        if idx == 0:
            assert best_model_path.endswith(f"epoch=0{idx}.ckpt")
        else:
            assert f"epoch={idx + 1}" in best_model_path


@pytest.mark.parametrize(['max_epochs', 'data_length'], [(1, 64), (2, 64), (3, 32)])
def test_lr_schedulers_step_count(tmpdir, max_epochs, data_length):
    """
    This test validates that checkpoint is always saved after lr_scheduler beeing updated during training
    """

    class TestModel(BoringModel):

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            lr_scheduler_dict = {'scheduler': lr_scheduler, 'interval': 'step'}
            return [optimizer], [lr_scheduler_dict]

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, data_length))

    train_step_checkpoint_callback = ModelCheckpoint(dirpath=f"{tmpdir}/every_train_step", every_n_train_steps=1)
    val_epoch_checkpoint_callback = ModelCheckpoint(dirpath=f"{tmpdir}/every_val_epoch", every_n_val_epochs=1)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=max_epochs,
        callbacks=[train_step_checkpoint_callback, val_epoch_checkpoint_callback]
    )
    trainer.fit(model)
    step_idx = data_length * max_epochs - 1
    train_step_lr_scheduler = pl_load(f"{tmpdir}/every_train_step/epoch={max_epochs-1}-step={step_idx}.ckpt"
                                      )['lr_schedulers'][0]
    val_epoch_lr_scheduler = pl_load(f"{tmpdir}/every_val_epoch/epoch={max_epochs-1}-step={step_idx}.ckpt"
                                     )['lr_schedulers'][0]
    #
    assert train_step_lr_scheduler['last_epoch'] == val_epoch_lr_scheduler['last_epoch'] == step_idx + 1
    assert train_step_lr_scheduler['_step_count'] == val_epoch_lr_scheduler['_step_count'] == step_idx + 2
