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
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import tests.helpers.pipelines as tpipes
import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core import memory
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel, RandomDataset
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel


class CustomClassificationModelDP(ClassificationModel):
    def _step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return {"logits": logits, "y": y}

    def training_step(self, batch, batch_idx):
        out = self._step(batch, batch_idx)
        loss = F.cross_entropy(out["logits"], out["y"])
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step_end(self, outputs):
        self.log("val_acc", self.valid_acc(outputs["logits"], outputs["y"]))

    def test_step_end(self, outputs):
        self.log("test_acc", self.test_acc(outputs["logits"], outputs["y"]))


@RunIf(min_gpus=2)
def test_multi_gpu_early_stop_dp(tmpdir):
    """Make sure DDP works. with early stopping"""
    tutils.set_random_master_port()

    dm = ClassifDataModule()
    model = CustomClassificationModelDP()

    trainer_options = dict(
        default_root_dir=tmpdir,
        callbacks=[EarlyStopping(monitor="val_acc")],
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        accelerator="dp",
    )

    tpipes.run_model_test(trainer_options, model, dm)


@RunIf(min_gpus=2)
def test_multi_gpu_model_dp(tmpdir):
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        accelerator="dp",
        progress_bar_refresh_rate=0,
    )

    model = BoringModel()

    tpipes.run_model_test(trainer_options, model)

    # test memory helper functions
    memory.get_memory_profile("min_max")


class ReductionTestModel(BoringModel):
    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=2)

    def add_outputs(self, output, device):
        output.update(
            {
                "reduce_int": torch.tensor(device.index, dtype=torch.int, device=device),
                "reduce_float": torch.tensor(device.index, dtype=torch.float, device=device),
            }
        )

    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        self.add_outputs(output, batch.device)
        return output

    def validation_step(self, batch, batch_idx):
        output = super().validation_step(batch, batch_idx)
        self.add_outputs(output, batch.device)
        return output

    def test_step(self, batch, batch_idx):
        output = super().test_step(batch, batch_idx)
        self.add_outputs(output, batch.device)
        return output

    def training_epoch_end(self, outputs):
        assert outputs[0]["loss"].shape == torch.Size([])
        assert outputs[0]["reduce_int"].item() == 0  # mean([0, 1]) = 0
        assert outputs[0]["reduce_float"].item() == 0.5  # mean([0., 1.]) = 0.5


def test_dp_raise_exception_with_batch_transfer_hooks(tmpdir, monkeypatch):
    """
    Test that an exception is raised when overriding batch_transfer_hooks in DP model.
    """
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)

    class CustomModel(BoringModel):
        def transfer_batch_to_device(self, batch, device):
            batch = batch.to(device)
            return batch

    trainer_options = dict(default_root_dir=tmpdir, max_steps=7, gpus=[0, 1], accelerator="dp")

    trainer = Trainer(**trainer_options)
    model = CustomModel()

    with pytest.raises(MisconfigurationException, match=r"Overriding `transfer_batch_to_device` is not .* in DP"):
        trainer.fit(model)

    class CustomModel(BoringModel):
        def on_before_batch_transfer(self, batch, dataloader_idx):
            batch += 1
            return batch

    trainer = Trainer(**trainer_options)
    model = CustomModel()

    with pytest.raises(MisconfigurationException, match=r"Overriding `on_before_batch_transfer` is not .* in DP"):
        trainer.fit(model)

    class CustomModel(BoringModel):
        def on_after_batch_transfer(self, batch, dataloader_idx):
            batch += 1
            return batch

    trainer = Trainer(**trainer_options)
    model = CustomModel()

    with pytest.raises(MisconfigurationException, match=r"Overriding `on_after_batch_transfer` is not .* in DP"):
        trainer.fit(model)


@RunIf(min_gpus=2)
def test_dp_training_step_dict(tmpdir):
    """This test verifies that dp properly reduces dictionaries"""
    model = ReductionTestModel()
    model.training_step_end = None
    model.validation_step_end = None
    model.test_step_end = None

    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        gpus=2,
        accelerator="dp",
    )
    trainer.fit(model)
