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
r"""
Timer
^^^^^
"""
import logging
from datetime import datetime, timedelta
from typing import Union

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info

log = logging.getLogger(__name__)


class Timer(Callback):
    """
    The Timer callback tracks the time spent in the training loop and interrupts the Trainer
    if the given time limit is reached.

    Args:
        duration: A string in the format HH:MM:SS (hours, minutes seconds), or a :class:`datetime.timedelta`.
            Mutually exclusive with arguments `hours`, `minutes`, etc.
        interval: Determines if the interruption happens on epoch level or mid-epoch.
            Can be either `epoch` or `step`.
        verbose: Set this to ``False`` to suppress logging messages.
    """

    INTERVAL_CHOICES = ("epoch", "step")

    def __init__(self, duration: Union[str, timedelta], interval: str = "epoch", verbose: bool = True):
        super().__init__()
        if isinstance(duration, str):
            hms = datetime.strptime(duration.strip(), "%H:%M:%S")
            duration = timedelta(hours=hms.hour, minutes=hms.minute, seconds=hms.second)
        self._duration = duration
        self._interval = interval
        self._verbose = verbose
        self._start_time = None

    @property
    def time_elapsed(self) -> timedelta:
        if self._start_time is None:
            return timedelta()
        return datetime.now() - self._start_time

    @property
    def time_remaining(self) -> timedelta:
        if self._start_time is None:
            return self._duration
        return self._duration - self.time_elapsed

    def on_fit_start(self, trainer, *args, **kwargs) -> None:
        self._start_time = datetime.now()

    def on_batch_end(self, trainer, *args, **kwargs) -> None:
        if self._interval != "step":
            return
        self._check_time_remaining(trainer)

    def on_epoch_end(self, trainer, *args, **kwargs) -> None:
        if self._interval != "epoch":
            return
        self._check_time_remaining(trainer)

    def _check_time_remaining(self, trainer) -> None:
        should_stop = self.time_elapsed >= self._duration
        should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop and self._verbose:
            rank_zero_info("Time limit reached. Signaling Trainer to stop.")
