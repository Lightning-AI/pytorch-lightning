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
import logging
from unittest.mock import Mock

import pytest
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops import _FitLoop


def test_outputs_format(tmp_path):
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

        def on_train_batch_end(self, outputs, *_):
            HookedModel._check_output(outputs)

    model = HookedModel()

    # fit model
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=2,
        limit_test_batches=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@pytest.mark.parametrize("seed_once", [True, False])
def test_training_starts_with_seed(tmp_path, seed_once):
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
        sequence0 = run_training(default_root_dir=tmp_path, max_steps=2, num_sanity_val_steps=0)
        sequence1 = run_training(default_root_dir=tmp_path, max_steps=2, num_sanity_val_steps=2)
        assert not torch.allclose(sequence0, sequence1)
    else:
        seed_everything(123)
        sequence0 = run_training(default_root_dir=tmp_path, max_steps=2, num_sanity_val_steps=0)
        seed_everything(123)
        sequence1 = run_training(default_root_dir=tmp_path, max_steps=2, num_sanity_val_steps=2)
        assert torch.allclose(sequence0, sequence1)


@pytest.mark.parametrize(("max_epochs", "batch_idx_"), [(2, 5), (3, 8), (4, 12)])
def test_on_train_batch_start_return_minus_one(max_epochs, batch_idx_, tmp_path):
    class CurrentModel(BoringModel):
        def on_train_batch_start(self, batch, batch_idx):
            if batch_idx == batch_idx_:
                return -1
            return None

    model = CurrentModel()
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=max_epochs, limit_train_batches=10)
    trainer.fit(model)
    if batch_idx_ > trainer.num_training_batches - 1:
        assert trainer.fit_loop.batch_idx == trainer.num_training_batches - 1
        assert trainer.global_step == trainer.num_training_batches * max_epochs
    else:
        assert trainer.fit_loop.batch_idx == batch_idx_
        assert trainer.global_step == batch_idx_ * max_epochs


def test_should_stop_mid_epoch(tmp_path):
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
    trainer = Trainer(default_root_dir=tmp_path, max_epochs=1, limit_train_batches=10, limit_val_batches=1)
    trainer.fit(model)

    # even though we stopped mid epoch, the fit loop finished normally and the current epoch was increased
    assert trainer.current_epoch == 1
    assert trainer.global_step == 5
    assert model.validation_called_at == (0, 5)


def test_fit_loop_done_log_messages(caplog):
    trainer = Mock(spec=Trainer)
    fit_loop = _FitLoop(trainer, max_epochs=1)

    trainer.should_stop = False
    fit_loop.max_batches = 5
    assert not fit_loop.done
    assert not caplog.messages

    fit_loop.max_batches = 0
    assert fit_loop.done
    assert "No training batches" in caplog.text
    caplog.clear()
    fit_loop.max_batches = 5

    epoch_loop = Mock()
    epoch_loop.global_step = 10
    fit_loop.epoch_loop = epoch_loop
    epoch_loop.max_steps = 10
    assert fit_loop.done
    assert "max_steps=10` reached" in caplog.text
    caplog.clear()
    epoch_loop.max_steps = 20

    fit_loop.epoch_progress.current.processed = 3
    fit_loop.max_epochs = 3
    trainer.should_stop = True
    assert fit_loop.done
    assert "max_epochs=3` reached" in caplog.text
    caplog.clear()
    fit_loop.max_epochs = 5

    fit_loop.epoch_loop.min_steps = 0
    with caplog.at_level(level=logging.DEBUG, logger="lightning.pytorch.utilities.rank_zero"):
        assert fit_loop.done
    assert "should_stop` was set" in caplog.text

    fit_loop.epoch_loop.min_steps = 100
    assert not fit_loop.done


@pytest.mark.parametrize(
    ("min_epochs", "min_steps", "current_epoch", "early_stop", "fit_loop_done", "raise_debug_msg"),
    [
        (4, None, 100, True, True, False),
        (4, None, 3, False, False, False),
        (4, 10, 3, False, False, False),
        (None, 10, 4, True, True, True),
        (4, None, 4, True, True, True),
        (4, 10, 4, True, True, True),
    ],
)
def test_should_stop_early_stopping_conditions_met(
    caplog, min_epochs, min_steps, current_epoch, early_stop, fit_loop_done, raise_debug_msg
):
    """Test that checks that debug message is logged when users sets `should_stop` and min conditions are met."""
    trainer = Trainer(min_epochs=min_epochs, min_steps=min_steps, limit_val_batches=0, max_epochs=100)
    trainer.fit_loop.max_batches = 10
    trainer.should_stop = True
    trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = (
        current_epoch * trainer.num_training_batches
    )
    trainer.fit_loop.epoch_loop.batch_progress.current.ready = 10
    trainer.fit_loop.epoch_progress.current.processed = current_epoch

    message = "`Trainer.fit` stopped: `trainer.should_stop` was set."
    with caplog.at_level(level=logging.DEBUG, logger="lightning.pytorch.utilities.rank_zero"):
        assert trainer.fit_loop.done is fit_loop_done

    assert (message in caplog.text) is raise_debug_msg
    assert trainer.fit_loop._can_stop_early is early_stop
