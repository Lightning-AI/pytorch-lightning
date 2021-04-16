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
import logging
import time
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


def test_trainer_flag(caplog):

    class TestModel(BoringModel):

        def on_fit_start(self):
            raise SystemExit()

    trainer = Trainer(max_time=dict(seconds=1337))
    with pytest.raises(SystemExit):
        trainer.fit(TestModel())
    timer = [c for c in trainer.callbacks if isinstance(c, Timer)][0]
    assert timer._duration == 1337

    trainer = Trainer(max_time=dict(seconds=1337), callbacks=[Timer()])
    with pytest.raises(SystemExit), caplog.at_level(level=logging.INFO):
        trainer.fit(TestModel())
    assert "callbacks list already contains a Timer" in caplog.text


@pytest.mark.parametrize(
    "duration,expected", [
        (None, None),
        ("00:00:00:22", timedelta(seconds=22)),
        ("12:34:56:65", timedelta(days=12, hours=34, minutes=56, seconds=65)),
        (timedelta(weeks=52, milliseconds=1), timedelta(weeks=52, milliseconds=1)),
        (dict(weeks=52, days=1), timedelta(weeks=52, days=1)),
    ]
)
def test_timer_parse_duration(duration, expected):
    timer = Timer(duration=duration)
    assert (timer.time_remaining() == expected is None) or (timer.time_remaining() == expected.total_seconds())


def test_timer_interval_choice():
    Timer(duration=timedelta(), interval="step")
    Timer(duration=timedelta(), interval="epoch")
    with pytest.raises(MisconfigurationException, match="Unsupported parameter value"):
        Timer(duration=timedelta(), interval="invalid")


@patch("pytorch_lightning.callbacks.timer.time")
def test_timer_time_remaining(time_mock):
    """ Test that the timer tracks the elapsed and remaining time correctly. """
    start_time = time.monotonic()
    duration = timedelta(seconds=10)
    time_mock.monotonic.return_value = start_time
    timer = Timer(duration=duration)
    assert timer.time_remaining() == duration.total_seconds()
    assert timer.time_elapsed() == 0

    # timer not started yet
    time_mock.monotonic.return_value = start_time + 60
    assert timer.start_time() is None
    assert timer.time_remaining() == 10
    assert timer.time_elapsed() == 0

    # start timer
    time_mock.monotonic.return_value = start_time
    timer.on_train_start(trainer=Mock(), pl_module=Mock())
    assert timer.start_time() == start_time

    # pretend time has elapsed
    elapsed = 3
    time_mock.monotonic.return_value = start_time + elapsed
    assert timer.start_time() == start_time
    assert round(timer.time_remaining()) == 7
    assert round(timer.time_elapsed()) == 3


def test_timer_stops_training(tmpdir):
    """ Test that the timer stops training before reaching max_epochs """
    model = BoringModel()
    duration = timedelta(milliseconds=100)
    timer = Timer(duration=duration)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1000,
        callbacks=[timer],
    )
    trainer.fit(model)
    assert trainer.global_step > 1
    assert trainer.current_epoch < 999


@pytest.mark.parametrize("interval", ["step", "epoch"])
def test_timer_zero_duration_stop(tmpdir, interval):
    """ Test that the timer stops training immediately after the first check occurs. """
    model = BoringModel()
    duration = timedelta(0)
    timer = Timer(duration=duration, interval=interval)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[timer],
    )
    trainer.fit(model)
    if interval == "step":
        # timer triggers stop on step end
        assert trainer.global_step == 1
        assert trainer.current_epoch == 0
    else:
        # timer triggers stop on epoch end
        assert trainer.global_step == len(trainer.train_dataloader)
        assert trainer.current_epoch == 0


@pytest.mark.parametrize("min_steps,min_epochs", [
    (None, 2),
    (3, None),
    (3, 2),
])
def test_timer_duration_min_steps_override(tmpdir, min_steps, min_epochs):
    model = BoringModel()
    duration = timedelta(0)
    timer = Timer(duration=duration)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[timer],
        min_steps=min_steps,
        min_epochs=min_epochs,
    )
    trainer.fit(model)
    if min_epochs:
        assert trainer.current_epoch >= min_epochs - 1
    if min_steps:
        assert trainer.global_step >= min_steps - 1
    assert timer.time_elapsed() > duration.total_seconds()


def test_timer_resume_training(tmpdir):
    """ Test that the timer can resume together with the Trainer. """
    model = BoringModel()
    timer = Timer(duration=timedelta(milliseconds=200))
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1)

    # initial training
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=100,
        callbacks=[timer, checkpoint_callback],
    )
    trainer.fit(model)
    assert not timer._offset
    assert timer.time_remaining() <= 0
    assert trainer.current_epoch < 99
    saved_global_step = trainer.global_step

    # resume training (with depleted timer
    timer = Timer(duration=timedelta(milliseconds=200))
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[timer, checkpoint_callback],
        resume_from_checkpoint=checkpoint_callback.best_model_path,
    )
    trainer.fit(model)
    assert timer._offset > 0
    assert trainer.global_step == saved_global_step + 1


@RunIf(skip_windows=True)
def test_timer_track_stages(tmpdir):
    """ Test that the timer tracks time also for other stages (train/val/test). """
    # note: skipped on windows because time resolution of time.monotonic() is not high enough for this fast test
    model = BoringModel()
    timer = Timer()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=5,
        callbacks=[timer],
    )
    trainer.fit(model)
    assert timer.time_elapsed() == timer.time_elapsed("train") > 0
    assert timer.time_elapsed("validate") > 0
    assert timer.time_elapsed("test") == 0
    trainer.test(model)
    assert timer.time_elapsed("test") > 0
