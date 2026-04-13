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
import re
import time
from datetime import timedelta
from unittest.mock import patch

import pytest
from torch.utils.data import DataLoader

from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.utilities.exceptions import MisconfigurationException


@pytest.mark.parametrize("max_epochs", [1, 2, 3])
@pytest.mark.parametrize("denominator", [1, 3, 4])
def test_val_check_interval(tmp_path, max_epochs, denominator):
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

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=False,
        max_epochs=max_epochs,
        val_check_interval=1 / denominator,
    )
    trainer.fit(model)

    assert model.train_epoch_calls == max_epochs
    assert model.val_epoch_calls == max_epochs * denominator


@pytest.mark.parametrize("value", [1, 1.0])
def test_val_check_interval_info_message(caplog, value):
    with caplog.at_level(logging.INFO):
        Trainer(val_check_interval=value)
    assert f"`Trainer(val_check_interval={value})` was configured" in caplog.text
    message = "configured so validation will run"
    assert message in caplog.text

    caplog.clear()

    # the message should not appear by default
    with caplog.at_level(logging.INFO):
        Trainer()
    assert message not in caplog.text


@pytest.mark.parametrize("use_infinite_dataset", [True, False])
@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
def test_validation_check_interval_exceed_data_length_correct(tmp_path, use_infinite_dataset, accumulate_grad_batches):
    data_samples_train = 4
    max_epochs = 3
    max_steps = data_samples_train * max_epochs
    max_opt_steps = max_steps // accumulate_grad_batches

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.validation_called_at_step = set()

        def validation_step(self, *args):
            self.validation_called_at_step.add(self.trainer.fit_loop.total_batch_idx + 1)
            return super().validation_step(*args)

        def train_dataloader(self):
            train_ds = (
                RandomIterableDataset(32, count=max_steps + 100)
                if use_infinite_dataset
                else RandomDataset(32, length=data_samples_train)
            )
            return DataLoader(train_ds)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_val_batches=1,
        max_steps=max_opt_steps,
        val_check_interval=3,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(model)

    assert trainer.current_epoch == 1 if use_infinite_dataset else max_epochs
    assert trainer.global_step == max_opt_steps
    assert sorted(model.validation_called_at_step) == [3, 6, 9, 12]


def test_validation_check_interval_exceed_data_length_wrong():
    trainer = Trainer(
        limit_train_batches=10,
        val_check_interval=100,
        logger=False,
        enable_checkpointing=False,
    )

    model = BoringModel()
    with pytest.raises(ValueError, match="must be less than or equal to the number of the training batches"):
        trainer.fit(model)


def test_val_check_interval_float_with_none_check_val_every_n_epoch():
    """Test that an exception is raised when `val_check_interval` is set to float with
    `check_val_every_n_epoch=None`"""
    with pytest.raises(
        MisconfigurationException,
        match=re.escape(
            "`val_check_interval` should be an integer or a time-based duration (str 'DD:HH:MM:SS', "
            "datetime.timedelta, or dict kwargs for timedelta) when `check_val_every_n_epoch=None`."
        ),
    ):
        Trainer(
            val_check_interval=0.5,
            check_val_every_n_epoch=None,
        )


@pytest.mark.parametrize(
    "interval",
    [
        "00:00:00:02",
        {"seconds": 2},
        timedelta(seconds=2),
    ],
)
def test_time_based_val_check_interval(tmp_path, interval):
    call_count = {"count": 0}

    def fake_time():
        result = call_count["count"]
        call_count["count"] += 2
        return result

    with patch("time.monotonic", side_effect=fake_time):
        trainer = Trainer(
            default_root_dir=tmp_path,
            logger=False,
            enable_checkpointing=False,
            max_epochs=1,
            max_steps=5,  # 5 steps: simulate 10s total wall-clock time
            limit_val_batches=1,
            val_check_interval=interval,  # every 2s
        )
        model = BoringModel()
        trainer.fit(model)

    # Assert 5 validations happened
    val_runs = trainer.fit_loop.epoch_loop.val_loop.batch_progress.total.completed
    # The number of validation runs should be equal to the number of times we called fake_time
    assert val_runs == 5, f"Expected 5 validations, got {val_runs}"


@pytest.mark.parametrize(
    ("check_val_every_n_epoch", "val_check_interval", "epoch_duration", "expected_val_batches", "description"),
    [
        (None, "00:00:00:04", 2, [0, 1, 0, 1, 0], "val_check_interval timer only, no epoch gating"),
        (1, "00:00:00:06", 8, [1, 1, 2, 1, 1], "val_check_interval timer only, no epoch gating"),
        (2, "00:00:00:06", 9, [0, 2, 0, 2, 0], "epoch gating, timer shorter than epoch"),
        (2, "00:00:00:03", 9, [0, 3, 0, 3, 0], "epoch gating, timer much shorter than epoch"),
        (2, "00:00:00:20", 9, [0, 0, 0, 1, 0], "epoch gating, timer longer than epoch"),
    ],
)
def test_time_and_epoch_gated_val_check(
    tmp_path, check_val_every_n_epoch, val_check_interval, epoch_duration, expected_val_batches, description
):
    call_count = {"count": 0}

    # Simulate time in steps (each batch is 1 second, epoch_duration=seconds per epoch)
    def fake_time():
        result = call_count["count"]
        call_count["count"] += 1
        return result

    # Custom model to record when validation happens (on what epoch)
    class TestModel(BoringModel):
        val_batches = []
        val_epoch_calls = 0

        def on_train_batch_end(self, *args, **kwargs):
            if (
                isinstance(self.trainer.check_val_every_n_epoch, int)
                and self.trainer.check_val_every_n_epoch > 1
                and (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch != 0
            ):
                time.monotonic()

        def on_train_epoch_end(self, *args, **kwargs):
            print(trainer.fit_loop.epoch_loop.val_loop.batch_progress.current.completed)
            self.val_batches.append(trainer.fit_loop.epoch_loop.val_loop.batch_progress.total.completed)

        def on_validation_epoch_start(self) -> None:
            self.val_epoch_calls += 1

    max_epochs = 5
    max_steps = max_epochs * epoch_duration
    limit_train_batches = epoch_duration

    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "logger": False,
        "enable_checkpointing": False,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "limit_val_batches": 1,
        "limit_train_batches": limit_train_batches,
        "val_check_interval": val_check_interval,
        "check_val_every_n_epoch": check_val_every_n_epoch,
    }

    with patch("time.monotonic", side_effect=fake_time):
        model = TestModel()
        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model)

    # Validate which epochs validation happened
    assert model.val_batches == expected_val_batches, (
        f"\nFAILED: {description}"
        f"\nExpected validation at batches: {expected_val_batches},"
        f"\nGot: {model.val_batches, model.val_epoch_calls}\n"
    )
