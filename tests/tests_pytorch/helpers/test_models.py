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

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from tests_pytorch.helpers.advanced_models import BasicGAN, ParityModuleMNIST, ParityModuleRNN
from tests_pytorch.helpers.datamodules import ClassifDataModule, RegressDataModule
from tests_pytorch.helpers.simple_models import ClassificationModel, RegressionModel


class AMPTestModel(BoringModel):
    def _step(self, batch):
        self._assert_autocast_enabled()
        output = self(batch)
        is_bfloat16 = self.trainer.precision_plugin.precision == "bf16"
        assert output.dtype == torch.float16 if not is_bfloat16 else torch.bfloat16
        loss = self.loss(batch, output)
        return loss

    def loss(self, batch, prediction):
        # todo (sean): convert bfloat16 to float32 as mse loss for cpu amp is currently not supported
        if self.trainer.precision_plugin.device == "cpu":
            prediction = prediction.float()
        return super().loss(batch, prediction)

    def training_step(self, batch, batch_idx):
        output = self._step(batch)
        return {"loss": output}

    def validation_step(self, batch, batch_idx):
        output = self._step(batch)
        return {"x": output}

    def test_step(self, batch, batch_idx):
        output = self._step(batch)
        return {"y": output}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self._assert_autocast_enabled()
        output = self(batch)
        is_bfloat16 = self.trainer.precision_plugin.precision == "bf16"
        assert output.dtype == torch.float16 if not is_bfloat16 else torch.bfloat16
        return output

    def _assert_autocast_enabled(self):
        if self.trainer.precision_plugin.device == "cpu":
            assert torch.is_autocast_cpu_enabled()
        else:
            assert torch.is_autocast_enabled()


@pytest.mark.parametrize(
    "data_class,model_class",
    [
        (None, BoringModel),
        (None, BasicGAN),
        (None, ParityModuleRNN),
        (None, ParityModuleMNIST),
        (ClassifDataModule, ClassificationModel),
        (RegressDataModule, RegressionModel),
    ],
)
def test_models(tmpdir, data_class, model_class):
    """Test simple models."""
    dm = data_class() if data_class else data_class
    model = model_class()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    trainer.fit(model, datamodule=dm)

    if dm is not None:
        trainer.test(model, datamodule=dm)

    model.to_torchscript()
    if data_class:
        model.to_onnx(os.path.join(tmpdir, "my-model.onnx"), input_sample=dm.sample)
