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

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import DummyException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class TrainerStagesErrorsModel(BoringModel):
    def on_train_start(self) -> None:
        raise DummyException("Error during train")

    def on_validation_start(self) -> None:
        raise DummyException("Error during validation")

    def on_test_start(self) -> None:
        raise DummyException("Error during test")

    def on_predict_start(self) -> None:
        raise DummyException("Error during predict")


@pytest.mark.parametrize(
    "accelerator,num_processes", [(None, 1), pytest.param("ddp_cpu", 2, marks=RunIf(skip_windows=True))]
)
def test_error_handling_all_stages(tmpdir, accelerator, num_processes):
    model = TrainerStagesErrorsModel()
    trainer = Trainer(default_root_dir=tmpdir, accelerator=accelerator, num_processes=num_processes, fast_dev_run=True)
    with pytest.raises(DummyException, match=r"Error during train"):
        trainer.fit(model)
    with pytest.raises(DummyException, match=r"Error during validation"):
        trainer.validate(model)
    with pytest.raises(DummyException, match=r"Error during test"):
        trainer.test(model)
    with pytest.raises(DummyException, match=r"Error during predict"):
        trainer.predict(model, model.val_dataloader())
