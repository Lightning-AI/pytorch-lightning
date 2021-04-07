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
from datetime import timedelta, datetime
from unittest.mock import Mock, patch

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


@pytest.mark.parametrize("duration,expected", [
    ("00:00:22", timedelta(seconds=22)),
    ("12:34:56", timedelta(hours=12, minutes=34, seconds=56)),
    (timedelta(weeks=52, milliseconds=1), timedelta(weeks=52, milliseconds=1)),
])
def test_timer_parse_duration(duration, expected):
    timer = Timer(duration=duration)
    assert timer.time_remaining == expected


def test_timer_interval_choice():
    Timer(duration=timedelta(), interval="step")
    Timer(duration=timedelta(), interval="epoch")
    with pytest.raises(MisconfigurationException, match="Unsupported parameter value"):
        Timer(duration=timedelta(), interval="invalid")


@patch("pytorch_lightning.callbacks.timer.datetime")
def test_timer_time_remaining(datetime_mock):
    """ Test that the timer tracks the elapsed and remaining time correctly. """
    start_time = datetime.now()
    duration = timedelta(seconds=10)
    datetime_mock.now.return_value = start_time
    timer = Timer(duration=duration)
    assert timer.time_remaining == duration
    assert timer.time_elapsed == timedelta(0)

    # timer not started yet
    datetime_mock.now.return_value = start_time + timedelta(minutes=1)
    assert timer.start_time is None
    assert timer.time_remaining == timedelta(seconds=10)
    assert timer.time_elapsed == timedelta(seconds=0)

    # start timer
    datetime_mock.now.return_value = start_time
    timer.on_train_start(trainer=Mock(), pl_module=Mock())
    assert timer.start_time == start_time

    # pretend time has elapsed
    elapsed = timedelta(seconds=3)
    datetime_mock.now.return_value = start_time + elapsed
    assert timer.start_time == start_time
    assert timer.time_remaining == timedelta(seconds=7)
    assert timer.time_elapsed == timedelta(seconds=3)


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
    assert trainer.current_epoch < 999
