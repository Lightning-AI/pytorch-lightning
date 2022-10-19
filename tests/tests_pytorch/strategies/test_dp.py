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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import tests_pytorch.helpers.pipelines as tpipes
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.demos.boring_classes import BoringModel, RandomDataset
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


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


@RunIf(min_cuda_gpus=2)
def test_multi_gpu_early_stop_dp(tmpdir):
    """Make sure DDP works.

    with early stopping
    """
    dm = ClassifDataModule()
    model = CustomClassificationModelDP()

    trainer_options = dict(
        default_root_dir=tmpdir,
        callbacks=[EarlyStopping(monitor="val_acc")],
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        accelerator="gpu",
        devices=[0, 1],
        strategy="dp",
    )

    tpipes.run_model_test(trainer_options, model, dm)


@RunIf(min_cuda_gpus=2)
def test_multi_gpu_model_dp(tmpdir):
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        accelerator="gpu",
        devices=[0, 1],
        strategy="dp",
        enable_progress_bar=False,
    )

    model = BoringModel()

    tpipes.run_model_test(trainer_options, model)


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
        self._assert_extra_outputs(outputs)

    def validation_epoch_end(self, outputs):
        assert outputs[0]["x"].shape == torch.Size([2])
        self._assert_extra_outputs(outputs)

    def test_epoch_end(self, outputs):
        assert outputs[0]["y"].shape == torch.Size([2])
        self._assert_extra_outputs(outputs)

    def _assert_extra_outputs(self, outputs):
        out = outputs[0]["reduce_int"]
        assert torch.eq(out, torch.tensor([0, 1], device="cuda:0")).all()
        assert out.dtype is torch.int

        out = outputs[0]["reduce_float"]
        assert torch.eq(out, torch.tensor([0.0, 1.0], device="cuda:0")).all()
        assert out.dtype is torch.float


@RunIf(min_cuda_gpus=2)
def test_dp_training_step_dict(tmpdir):
    """This test verifies that dp properly reduces dictionaries."""
    model = ReductionTestModel()
    model.training_step_end = None
    model.validation_step_end = None
    model.test_step_end = None

    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator="gpu",
        devices=2,
        strategy="dp",
    )
    trainer.fit(model)
    trainer.test(model)


@RunIf(min_cuda_gpus=2)
def test_dp_batch_not_moved_to_device_explicitly(tmpdir):
    """Test that with DP, batch is not moved to the device explicitly."""

    class CustomModel(BoringModel):
        def on_train_batch_start(self, batch, *args, **kargs):
            assert not batch.is_cuda

        def training_step(self, batch, batch_idx):
            assert batch.is_cuda
            return super().training_step(batch, batch_idx)

    trainer = pl.Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator="gpu",
        devices=2,
        strategy="dp",
    )

    trainer.fit(CustomModel())
