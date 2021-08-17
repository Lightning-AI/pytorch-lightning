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
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class TrainerStagesErrorsModel(BoringModel):
    def on_train_start(self) -> None:
        raise Exception("Error during train")

    def on_validation_start(self) -> None:
        raise Exception("Error during validation")

    def on_test_start(self) -> None:
        raise Exception("Error during test")

    def on_predict_start(self) -> None:
        raise Exception("Error during predict")


@pytest.mark.parametrize(
    "accelerator,num_processes",
    [
        (None, 1),
        pytest.param("ddp_cpu", 2, marks=RunIf(skip_windows=True)),
        pytest.param("ddp", 2, marks=RunIf(skip_windows=True, special=True)),
    ],
)
def test_error_handling_all_stages(tmpdir, accelerator, num_processes):
    model = TrainerStagesErrorsModel()
    trainer = Trainer(default_root_dir=tmpdir, accelerator=accelerator, num_processes=num_processes, fast_dev_run=True)
    with pytest.raises(Exception, match=r"Error during train"), mock.patch("pytorch_lightning.Trainer._on_exception"):
        trainer.fit(model)
    with pytest.raises(Exception, match=r"Error during validation"), mock.patch(
        "pytorch_lightning.Trainer._on_exception"
    ):
        trainer.validate(model)
    with pytest.raises(Exception, match=r"Error during test"), mock.patch("pytorch_lightning.Trainer._on_exception"):
        trainer.test(model)
    with pytest.raises(Exception, match=r"Error during predict"), mock.patch("pytorch_lightning.Trainer._on_exception"):
        trainer.predict(model, model.val_dataloader(), return_predictions=False)
