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

from pytorch_lightning import Trainer
from tests.helpers.zoo_data_models import ClassifDataModule, RegressDataModule, RegressionModel, ClassificationModel


@pytest.mark.parametrize("data_class,model_class", [(ClassifDataModule, ClassificationModel), (RegressDataModule, RegressionModel)])
def test_models(tmpdir, data_class, model_class):
    """Test simple models"""
    dm = data_class()
    model = model_class()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=5)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    model.to_torchscript()
    model.to_onnx(os.path.join(tmpdir, 'my-model.onnx'), input_sample=dm.sample)
