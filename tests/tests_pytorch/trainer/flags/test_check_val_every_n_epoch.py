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
import pytest
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.trainer.trainer import Trainer
from torch.utils.data import DataLoader


@pytest.mark.parametrize(
    ("max_epochs", "expected_val_loop_calls", "expected_val_batches"),
    [(1, 0, [0]), (4, 2, [0, 2, 0, 2]), (5, 2, [0, 2, 0, 2, 0])],
)
def test_check_val_every_n_epoch(tmpdir, max_epochs, expected_val_loop_calls, expected_val_batches):
    class TestModel(BoringModel):
        val_epoch_calls = 0
        val_batches = []

        def on_train_epoch_end(self, *args, **kwargs):
            self.val_batches.append(self.trainer.progress_bar_callback.total_val_batches)

        def on_validation_epoch_start(self) -> None:
            self.val_epoch_calls += 1

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        limit_val_batches=2,
        check_val_every_n_epoch=2,
        logger=False,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    assert model.val_epoch_calls == expected_val_loop_calls
    assert model.val_batches == expected_val_batches


def test_check_val_every_n_epoch_with_max_steps(tmpdir):
    data_samples_train = 2
    check_val_every_n_epoch = 3
    max_epochs = 4

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.validation_called_at_step = set()

        def validation_step(self, *args):
            self.validation_called_at_step.add(self.global_step)
            return super().validation_step(*args)

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, data_samples_train))

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=data_samples_train * max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,
    )

    trainer.fit(model)

    assert trainer.current_epoch == max_epochs
    assert trainer.global_step == max_epochs * data_samples_train
    assert list(model.validation_called_at_step) == [data_samples_train * check_val_every_n_epoch]
