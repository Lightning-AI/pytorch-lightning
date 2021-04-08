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
from typing import Any, Dict, Union

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.distributed import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)


class Interval(LightningEnum):
    step = "step"
    epoch = "epoch"


class Timer(Callback):
    """
    The Timer callback tracks the time spent in the training loop and interrupts the Trainer
    if the given time limit is reached.

    Args:
        duration: A string in the format DD:HH:MM:SS (days, hours, minutes seconds), or a :class:`datetime.timedelta`.
        interval: Determines if the interruption happens on epoch level or mid-epoch.
            Can be either `epoch` or `step`.
        verbose: Set this to ``False`` to suppress logging messages.

    Raises:
        MisconfigurationException:
            If ``interval`` is not one of the supported choices.

    Example::
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Timer

        # stop training after 12 hours
        timer = Timer(duration="00:12:00:00")

        # or provide a datetime.timedelta
        from datetime import timedelta
        timer = Timer(duration=timedelta(weeks=1))

        # force training to stop after given time limit
        trainer = Trainer(callbacks=[timer])
    """

    def __init__(self, duration: Union[str, timedelta], interval: str = Interval.step, verbose: bool = True):
        super().__init__()
        if isinstance(duration, str):
            dhms = duration.strip().split(":")
            dhms = [int(i) for i in dhms]
            duration = timedelta(days=dhms[0], hours=dhms[1], minutes=dhms[2], seconds=dhms[3])
        if interval not in set(Interval):
            raise MisconfigurationException(
                f"Unsupported parameter value `Timer(interval={interval})`. Possible choices are:"
                f" {', '.join(set(Interval))}"
            )
        self._duration = duration
        self._interval = interval
        self._verbose = verbose
        self._start_time = None
        self._offset = timedelta()

    @property
    def start_time(self):
        return self._start_time

    @property
    def time_elapsed(self) -> timedelta:
        if self._start_time is None:
            return self._offset
        return datetime.now() - self._start_time + self._offset

    @property
    def time_remaining(self) -> timedelta:
        return self._duration - self.time_elapsed

    def on_train_start(self, trainer, *args, **kwargs) -> None:
        self._start_time = datetime.now()

    def on_train_batch_end(self, trainer, *args, **kwargs) -> None:
        if self._interval != Interval.step:
            return
        self._check_time_remaining(trainer)

    def on_train_epoch_end(self, trainer, *args, **kwargs) -> None:
        if self._interval != Interval.epoch:
            return
        self._check_time_remaining(trainer)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "time_elapsed": self.time_elapsed,
        }

    def on_load_checkpoint(self, callback_state: Dict[str, Any]):
        self._offset = callback_state.get("time_elapsed", timedelta())

    def _check_time_remaining(self, trainer) -> None:
        should_stop = self.time_elapsed >= self._duration
        should_stop = trainer.accelerator.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop and self._verbose:
            rank_zero_info(
                f"Time limit reached. Elapsed time is {self.time_elapsed}."
                " Signaling Trainer to stop."
            )
