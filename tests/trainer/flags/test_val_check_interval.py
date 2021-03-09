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

from pytorch_lightning.trainer import Trainer
from tests.helpers import BoringModel


class TestModel(BoringModel):

    def __init__(self):
        super().__init__()
        self.train_epoch_calls = 0
        self.val_epoch_calls = 0

    def on_train_epoch_start(self) -> None:
        self.train_epoch_calls += 1

    def on_validation_epoch_start(self) -> None:
        if not self.trainer.sanity_checking:
            self.val_epoch_calls += 1


@pytest.mark.parametrize('max_epochs', [1, 2, 3])
@pytest.mark.parametrize('denominator', [1, 3, 4])
def test_val_check_interval(tmpdir, max_epochs, denominator):

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=max_epochs,
        val_check_interval=1 / denominator,
        logger=False,
    )
    trainer.fit(model)

    assert model.train_epoch_calls == max_epochs
    assert model.val_epoch_calls == max_epochs * denominator


@pytest.mark.parametrize('steps', [10, 100, 150])
def test_val_check_interval_steps(tmpdir, steps):

    max_epochs = 4
    model = TestModel()
    train_data_length = len(model.train_dataloader())

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=max_epochs,
        val_check_interval=steps,
        logger=False,
    )
    trainer.fit(model)

    assert model.train_epoch_calls == max_epochs
    assert model.val_epoch_calls == max_epochs * train_data_length // steps
