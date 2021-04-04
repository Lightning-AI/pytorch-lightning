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
from datetime import timedelta
from unittest import mock
from unittest.mock import ANY, call, MagicMock, Mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from tests.helpers import BoringModel


@pytest.mark.parametrize("duration,expected", [
    ("00:00:22", timedelta(seconds=22)),
    ("12:34:56", timedelta(hours=12, minutes=34, seconds=56)),
    (timedelta(weeks=52, milliseconds=1), timedelta(weeks=52, milliseconds=1)),
])
def test_timer_parse_duration(duration, expected):
    timer = Timer(duration=duration)
    assert timer.time_remaining == expected


