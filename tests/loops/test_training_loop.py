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
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel, RandomDataset
from tests.helpers.boring_model import RandomIterableDataset


def test_outputs_format(tmpdir):
    """Tests that outputs objects passed to model hooks and methods are consistent and in the correct format."""

    class HookedModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            self.log("foo", 123)
            output["foo"] = 123
            return output

        @staticmethod
        def _check_output(output):
            assert "loss" in output
            assert "foo" in output
            assert output["foo"] == 123

        def on_train_batch_end(self, outputs, batch, batch_idx):
            HookedModel._check_output(outputs)
            super().on_train_batch_end(outputs, batch, batch_idx)

        def training_epoch_end(self, outputs):
            assert len(outputs) == 2
            [HookedModel._check_output(output) for output in outputs]
            super().training_epoch_end(outputs)

    model = HookedModel()

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=2,
        limit_test_batches=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@pytest.mark.parametrize("seed_once", (True, False))
def test_training_starts_with_seed(tmpdir, seed_once):
    """Test the behavior of seed_everything on subsequent Trainer runs in combination with different settings of
    num_sanity_val_steps (which must not affect the random state)."""

    class SeededModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.seen_batches = []

        def training_step(self, batch, batch_idx):
            self.seen_batches.append(batch.view(-1))
            return super().training_step(batch, batch_idx)

    def run_training(**trainer_kwargs):
        model = SeededModel()
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model)
        return torch.cat(model.seen_batches)

    if seed_once:
        seed_everything(123)
        sequence0 = run_training(default_root_dir=tmpdir, max_steps=2, num_sanity_val_steps=0)
        sequence1 = run_training(default_root_dir=tmpdir, max_steps=2, num_sanity_val_steps=2)
        assert not torch.allclose(sequence0, sequence1)
    else:
        seed_everything(123)
        sequence0 = run_training(default_root_dir=tmpdir, max_steps=2, num_sanity_val_steps=0)
        seed_everything(123)
        sequence1 = run_training(default_root_dir=tmpdir, max_steps=2, num_sanity_val_steps=2)
        assert torch.allclose(sequence0, sequence1)


@pytest.mark.parametrize(["max_epochs", "batch_idx_"], [(2, 5), (3, 8), (4, 12)])
def test_on_train_batch_start_return_minus_one(max_epochs, batch_idx_, tmpdir):
    class CurrentModel(BoringModel):
        def on_train_batch_start(self, batch, batch_idx):
            if batch_idx == batch_idx_:
                return -1

    model = CurrentModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=max_epochs, limit_train_batches=10)
    trainer.fit(model)
    if batch_idx_ > trainer.num_training_batches - 1:
        assert trainer.fit_loop.batch_idx == trainer.num_training_batches - 1
        assert trainer.global_step == trainer.num_training_batches * max_epochs
    else:
        assert trainer.fit_loop.batch_idx == batch_idx_
        assert trainer.global_step == batch_idx_ * max_epochs


def test_should_stop_mid_epoch(tmpdir):
    """Test that training correctly stops mid epoch and that validation is still called at the right time."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.validation_called_at = None

        def training_step(self, batch, batch_idx):
            if batch_idx == 4:
                self.trainer.should_stop = True
            return super().training_step(batch, batch_idx)

        def validation_step(self, *args):
            self.validation_called_at = (self.trainer.current_epoch, self.trainer.global_step)
            return super().validation_step(*args)

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=10, limit_val_batches=1)
    trainer.fit(model)

    # even though we stopped mid epoch, the fit loop finished normally and the current epoch was increased
    assert trainer.current_epoch == 1
    assert trainer.global_step == 5
    assert model.validation_called_at == (0, 5)


def test_warning_valid_train_step_end(tmpdir):
    class ValidTrainStepEndModel(BoringModel):
        def training_step(self, batch, batch_idx):
            output = self(batch)
            return {"output": output, "batch": batch}

        def training_step_end(self, outputs):
            loss = self.loss(outputs["batch"], outputs["output"])
            return loss

    # No error is raised
    model = ValidTrainStepEndModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)

    trainer.fit(model)


@pytest.mark.parametrize("use_infinite_dataset", [True, False])
def test_validation_check_interval_exceed_data_length_correct(tmpdir, use_infinite_dataset):
    batch_size = 32
    data_samples_train = 10
    data_samples_val = 1

    if use_infinite_dataset:
        train_ds = RandomIterableDataset(size=batch_size, count=2_400_000_000)  # approx inf
    else:
        train_ds = RandomDataset(size=batch_size, length=data_samples_train)

    val_ds = RandomDataset(batch_size, data_samples_val)

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.validation_called_at_step = set()

        def training_step(self, batch, batch_idx):
            return super().training_step(batch, batch_idx)

        def validation_step(self, *args):
            self.validation_called_at_step.add(int(self.trainer.global_step))
            return super().validation_step(*args)

        def train_dataloader(self):
            return DataLoader(train_ds)

        def val_dataloader(self):
            return DataLoader(val_ds)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=data_samples_train * 3,
        val_check_interval=15,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )

    trainer.fit(model)

    # with a data length of 10 (or infinite), a val_check_interval of 15, and max_steps=30,
    # we should have validated twice
    if use_infinite_dataset:
        assert trainer.current_epoch == 1
    else:
        assert trainer.current_epoch == 3

    assert trainer.global_step == 30
    assert sorted(list(model.validation_called_at_step)) == [15, 30]


def test_validation_check_interval_exceed_data_length_wrong(tmpdir):
    model = BoringModel()

    with pytest.raises(ValueError):
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_steps=200,
            val_check_interval=100,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
        )
        trainer.fit(model)


def test_validation_check_interval_float_wrong(tmpdir):
    model = BoringModel()

    with pytest.raises(MisconfigurationException):
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_steps=200,
            val_check_interval=0.5,
            check_val_every_n_epoch=None,
            num_sanity_val_steps=0,
        )
        trainer.fit(model)


def test_validation_loop_every_5_epochs(tmpdir):
    batch_size = 32
    data_samples_train = 10
    data_samples_val = 1

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.validation_called_at_step = set()

        def training_step(self, batch, batch_idx):
            return super().training_step(batch, batch_idx)

        def validation_step(self, *args):
            self.validation_called_at_step.add(int(self.trainer.global_step))
            return super().validation_step(*args)

        def train_dataloader(self):
            return DataLoader(RandomDataset(batch_size, data_samples_train))

        def val_dataloader(self):
            return DataLoader(RandomDataset(batch_size, data_samples_val))

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=data_samples_train * 9,
        check_val_every_n_epoch=5,
        num_sanity_val_steps=0,
    )

    trainer.fit(model)

    # with a data length of 10, validation every 5 epochs, and max_steps=90, we should
    # validate once
    assert trainer.current_epoch == 9
    assert trainer.global_step == 90
    assert list(model.validation_called_at_step) == [50]
