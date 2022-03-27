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
import time
from datetime import timedelta
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_info

log = logging.getLogger(__name__)


class Interval(LightningEnum):
    step = "step"
    epoch = "epoch"


class Timer(Callback):
    """The Timer callback tracks the time spent in the training, validation, and test loops and interrupts the
    Trainer if the given time limit for the training loop is reached.

    Args:
        duration: A string in the format DD:HH:MM:SS (days, hours, minutes seconds), or a :class:`datetime.timedelta`,
            or a dict containing key-value compatible with :class:`~datetime.timedelta`.
        interval: Determines if the interruption happens on epoch level or mid-epoch.
            Can be either ``"epoch"`` or ``"step"``.
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

        # query training/validation/test time (in seconds)
        timer.time_elapsed("train")
        timer.start_time("validate")
        timer.end_time("test")
    """

    def __init__(
        self,
        duration: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        interval: str = Interval.step,
        verbose: bool = True,
    ) -> None:
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
        self._duration = duration.total_seconds() if duration is not None else None
        self._interval = interval
        self._verbose = verbose
        self._start_time: Dict[RunningStage, Optional[float]] = {stage: None for stage in RunningStage}
        self._end_time: Dict[RunningStage, Optional[float]] = {stage: None for stage in RunningStage}
        self._offset = 0

    def start_time(self, stage: str = RunningStage.TRAINING) -> Optional[float]:
        """Return the start time of a particular stage (in seconds)"""
        stage = RunningStage(stage)
        return self._start_time[stage]

    def end_time(self, stage: str = RunningStage.TRAINING) -> Optional[float]:
        """Return the end time of a particular stage (in seconds)"""
        stage = RunningStage(stage)
        return self._end_time[stage]

    def time_elapsed(self, stage: str = RunningStage.TRAINING) -> float:
        """Return the time elapsed for a particular stage (in seconds)"""
        start = self.start_time(stage)
        end = self.end_time(stage)
        offset = self._offset if stage == RunningStage.TRAINING else 0
        if start is None:
            return offset
        if end is None:
            return time.monotonic() - start + offset
        return end - start + offset

    def time_remaining(self, stage: str = RunningStage.TRAINING) -> Optional[float]:
        """Return the time remaining for a particular stage (in seconds)"""
        if self._duration is not None:
            return self._duration - self.time_elapsed(stage)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._start_time[RunningStage.TRAINING] = time.monotonic()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._end_time[RunningStage.TRAINING] = time.monotonic()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._start_time[RunningStage.VALIDATING] = time.monotonic()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._end_time[RunningStage.VALIDATING] = time.monotonic()

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._start_time[RunningStage.TESTING] = time.monotonic()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._end_time[RunningStage.TESTING] = time.monotonic()

    def on_fit_start(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        # this checks the time after the state is reloaded, regardless of the interval.
        # this is necessary in case we load a state whose timer is already depleted
        if self._duration is None:
            return
        self._check_time_remaining(trainer)

    def on_train_batch_end(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        if self._interval != Interval.step or self._duration is None:
            return
        self._check_time_remaining(trainer)

    def on_train_epoch_end(self, trainer: "pl.Trainer", *args: Any, **kwargs: Any) -> None:
        if self._interval != Interval.epoch or self._duration is None:
            return
        self._check_time_remaining(trainer)

    def state_dict(self) -> Dict[str, Any]:
        return {"time_elapsed": {stage.value: self.time_elapsed(stage) for stage in list(RunningStage)}}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        time_elapsed = state_dict.get("time_elapsed", {})
        self._offset = time_elapsed.get(RunningStage.TRAINING.value, 0)

    def _check_time_remaining(self, trainer: "pl.Trainer") -> None:
        assert self._duration is not None
        should_stop = self.time_elapsed() >= self._duration
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop and self._verbose:
            elapsed = timedelta(seconds=int(self.time_elapsed(RunningStage.TRAINING)))
            rank_zero_info(f"Time limit reached. Elapsed time is {elapsed}. Signaling Trainer to stop.")
