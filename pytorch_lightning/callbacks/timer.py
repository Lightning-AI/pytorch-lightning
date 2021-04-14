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
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Union, Optional

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.states import RunningStage
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
        duration: A string in the format DD:HH:MM:SS (days, hours, minutes seconds), or a :class:`datetime.timedelta`,
            or a dict containing key-value compatible with :class:`~datetime.timedelta`.
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

        # or provide a dictionary
        timer = Timer(duration=dict(weeks=4, days=2))

        # force training to stop after given time limit
        trainer = Trainer(callbacks=[timer])

        # query training/validation/test time
        timer.time_elapsed("train")
        timer.start_time("validate")
        timer.end_time("test")
    """

    def __init__(
        self,
        duration: Optional[Union[str, timedelta, Dict[str, int]]],
        interval: str = Interval.step,
        verbose: bool = True,
    ):
        super().__init__()
        if isinstance(duration, str):
            dhms = duration.strip().split(":")
            dhms = [int(i) for i in dhms]
            duration = timedelta(days=dhms[0], hours=dhms[1], minutes=dhms[2], seconds=dhms[3])
        if isinstance(duration, dict):
            duration = timedelta(**duration)
        if interval not in set(Interval):
            raise MisconfigurationException(
                f"Unsupported parameter value `Timer(interval={interval})`. Possible choices are:"
                f" {', '.join(set(Interval))}"
            )
        self._duration = duration
        self._interval = interval
        self._verbose = verbose
        self._start_time = defaultdict(lambda: None)
        self._end_time = defaultdict(lambda: None)
        self._offset = timedelta()

    def start_time(self, stage: str = RunningStage.TRAINING.value) -> Optional[datetime]:
        return self._start_time[stage]

    def end_time(self, stage: str = RunningStage.TRAINING.value) -> Optional[datetime]:
        return self._end_time[stage]

    def time_elapsed(self, stage: str = RunningStage.TRAINING.value) -> timedelta:
        start = self.start_time(stage)
        end = self.end_time(stage)
        offset = self._offset if stage == RunningStage.TRAINING else timedelta(0)
        if start is None:
            return offset
        if end is None:
            return datetime.now() - start + offset
        return end - start + offset

    def time_remaining(self, stage: str = RunningStage.TRAINING.value) -> Optional[timedelta]:
        if self._duration is not None:
            return self._duration - self.time_elapsed(stage)

    def on_train_start(self, *args, **kwargs) -> None:
        self._start_time.update({RunningStage.TRAINING.value: datetime.now()})

    def on_train_end(self, *args, **kwargs) -> None:
        self._end_time.update({RunningStage.TRAINING.value: datetime.now()})

    def on_validation_start(self, *args, **kwargs) -> None:
        self._start_time.update({RunningStage.VALIDATING.value: datetime.now()})

    def on_validation_end(self, *args, **kwargs) -> None:
        self._end_time.update({RunningStage.VALIDATING.value: datetime.now()})

    def on_test_start(self, *args, **kwargs) -> None:
        self._start_time.update({RunningStage.TESTING.value: datetime.now()})

    def on_test_end(self, *args, **kwargs) -> None:
        self._end_time.update({RunningStage.TESTING.value: datetime.now()})

    def on_train_batch_end(self, trainer, *args, **kwargs) -> None:
        if self._interval != Interval.step and self._duration is not None:
            return
        self._check_time_remaining(trainer)

    def on_train_epoch_end(self, trainer, *args, **kwargs) -> None:
        if self._interval != Interval.epoch and self._duration is not None:
            return
        self._check_time_remaining(trainer)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "time_elapsed": {k.value: self.time_elapsed(k.value) for k in RunningStage}
        }

    def on_load_checkpoint(self, callback_state: Dict[str, Any]):
        time_elapsed = callback_state.get("time_elapsed", defaultdict(timedelta))
        self._offset = time_elapsed[RunningStage.TRAINING.value]

    def _check_time_remaining(self, trainer) -> None:
        should_stop = self.time_elapsed() >= self._duration
        should_stop = trainer.accelerator.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop and self._verbose:
            rank_zero_info(f"Time limit reached. Elapsed time is {self.time_elapsed}. Signaling Trainer to stop.")
