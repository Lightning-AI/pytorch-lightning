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
import re

import pytest
import torch

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


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

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            HookedModel._check_output(outputs)
            super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

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
        progress_bar_refresh_rate=0,
        weights_summary=None,
    )
    trainer.fit(model)


def test_training_starts_with_seed(tmpdir):
    """ Test that the training always starts with the same random state (when using seed_everything). """

    class SeededModel(BoringModel):

        def __init__(self):
            super().__init__()
            self.seen_batches = []

        def training_step(self, batch, batch_idx):
            self.seen_batches.append(batch.view(-1))
            return super().training_step(batch, batch_idx)

    def run_training(**trainer_kwargs):
        model = SeededModel()
        seed_everything(123)
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model)
        return torch.cat(model.seen_batches)

    sequence0 = run_training(
        default_root_dir=tmpdir,
        max_steps=2,
        num_sanity_val_steps=0,
    )
    sequence1 = run_training(
        default_root_dir=tmpdir,
        max_steps=2,
        num_sanity_val_steps=2,
    )
    assert torch.allclose(sequence0, sequence1)


@pytest.mark.parametrize(['max_epochs', 'batch_idx_'], [(2, 5), (3, 8), (4, 12)])
def test_on_train_batch_start_return_minus_one(max_epochs, batch_idx_):

    class CurrentModel(BoringModel):

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            if batch_idx == batch_idx_:
                return -1

    model = CurrentModel()
    trainer = Trainer(max_epochs=max_epochs, limit_train_batches=10)
    trainer.fit(model)
    if batch_idx_ > trainer.num_training_batches - 1:
        assert trainer.train_loop.batch_idx == trainer.num_training_batches - 1
        assert trainer.global_step == trainer.num_training_batches * max_epochs
    else:
        assert trainer.train_loop.batch_idx == batch_idx_
        assert trainer.global_step == batch_idx_ * max_epochs


def test_should_stop_mid_epoch(tmpdir):
    """Test that training correctly stops mid epoch and that validation is still called at the right time"""

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
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=1,
    )
    trainer.fit(model)

    assert trainer.current_epoch == 0
    assert trainer.global_step == 5
    assert model.validation_called_at == (0, 4)


@pytest.mark.parametrize(['output'], [(5., ), ({'a': 5}, )])
def test_warning_invalid_trainstep_output(tmpdir, output):

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            return output

    model = TestModel()

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=1)
    with pytest.raises(
        MisconfigurationException,
        match=re.escape(
            "In automatic optimization, `training_step` must either return a Tensor, "
            "a dict with key 'loss' or None (where the step will be skipped)."
        )
    ):
        trainer.fit(model)
