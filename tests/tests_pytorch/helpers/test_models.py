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
import os

import pytest
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

from tests_pytorch.helpers.advanced_models import BasicGAN, ParityModuleMNIST, ParityModuleRNN
from tests_pytorch.helpers.datamodules import ClassifDataModule, RegressDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel, RegressionModel


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    ("data_class", "model_class"),
    [
        (None, BoringModel),
        pytest.param(None, BasicGAN, marks=RunIf(mps=False)),
        (None, ParityModuleRNN),
        (None, ParityModuleMNIST),
        pytest.param(ClassifDataModule, ClassificationModel, marks=RunIf(sklearn=True, onnx=True)),
        pytest.param(RegressDataModule, RegressionModel, marks=RunIf(sklearn=True, onnx=True)),
    ],
)
def test_models(tmp_path, data_class, model_class):
    """Test simple models."""
    dm = data_class() if data_class else data_class
    model = model_class()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1)

    trainer.fit(model, datamodule=dm)

    if dm is not None:
        trainer.test(model, datamodule=dm)

    model.to_torchscript()
    if data_class:
        model.to_onnx(os.path.join(tmp_path, "my-model.onnx"), input_sample=dm.sample)
