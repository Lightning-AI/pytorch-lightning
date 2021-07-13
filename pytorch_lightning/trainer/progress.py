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
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class BaseProgress:

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "BaseProgress":
        obj = cls()
        obj.load_state_dict(state_dict)
        return obj


@dataclass
class Tracker(BaseProgress):
    """
    Track an event's progress.

    Args:
        ready: Intended to track the number of events ready to start.
        started: Intended to be incremented after the event is started (e.g. after ``on_*_start`` runs).
        processed: Intended to be incremented after the event is processed.
        completed: Intended to be incremented after the event completes (e.g. after ``on_*_end`` runs).

    Attributes set to ``None`` are treated as unused and are restricted.
    """

    ready: Optional[int] = 0
    started: Optional[int] = 0
    processed: Optional[int] = 0
    completed: Optional[int] = 0

    def reset(self) -> None:
        if self.ready is not None:
            self.ready = 0
        if self.started is not None:
            self.started = 0
        if self.processed is not None:
            self.processed = 0
        if self.completed is not None:
            self.completed = 0

    def __setattr__(self, key: str, value: int) -> None:
        if getattr(self, key, 0) is None:
            raise AttributeError(f"The '{key}' attribute is meant to be unused")
        return super().__setattr__(key, value)

    def __repr__(self):
        # hide `None` fields
        args = [f"{k}={v}" for k, v in self.__dict__.items() if v is not None]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def reset_on_restart(self):
        """Reset the progress on restart"""
        value = self.completed if self.processed is None else self.processed

        if self.ready is not None:
            self.ready = value
        if self.started is not None:
            self.started = value
        if self.processed is not None:
            self.processed = value
        if self.completed is not None:
            self.completed = value


@dataclass
class Progress(BaseProgress):
    """
    Track aggregated and current progress.

    Args:
        total: Intended to track the total progress of an event
        current: Intended to track the current progress of an event
    """

    total: Tracker = field(default_factory=Tracker)
    current: Tracker = field(default_factory=Tracker)

    def increment_ready(self) -> None:
        if self.total.ready is None or self.current.ready is None:
            return
        self.total.ready += 1
        self.current.ready += 1

    def increment_started(self) -> None:
        if self.total.started is None or self.current.started is None:
            return
        self.total.started += 1
        self.current.started += 1

    def increment_processed(self) -> None:
        if self.total.processed is None or self.current.processed is None:
            return
        self.total.processed += 1
        self.current.processed += 1

    def increment_completed(self) -> None:
        if self.total.completed is None or self.current.completed is None:
            return
        self.total.completed += 1
        self.current.completed += 1

    @classmethod
    def from_defaults(cls, **kwargs: Optional[int]) -> "Progress":
        return cls(total=Tracker(**kwargs), current=Tracker(**kwargs))

    def load_state_dict(self, state_dict: dict) -> None:
        self.total.load_state_dict(state_dict["total"])
        self.current.load_state_dict(state_dict["current"])


class BatchProgress(Progress):
    """
    Tracks the batch progress

    Args:
        total: Tracks the total epoch progress
        current: Tracks the current epoch progress
    """


@dataclass
class EpochProgress(Progress):
    """
    Tracks the epoch progress
    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total epoch progress
        current: Tracks the current epoch progress
        batch: Tracks batch progress.
    """

    batch: BatchProgress = field(default_factory=BatchProgress)

    def reset_on_epoch(self) -> None:
        self.batch.current.reset()

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.batch.load_state_dict(state_dict["batch"])


@dataclass
class OptimizerProgress(BaseProgress):
    """
    Track optimizer progress.

    Args:
        step: Tracks ``optimizer.step`` calls.
        zero_grad: Tracks ``optimizer.zero_grad`` calls.
    """

    step: Progress = field(default_factory=lambda: Progress.from_defaults(processed=None))
    zero_grad: Progress = field(default_factory=lambda: Progress.from_defaults(processed=None))

    def reset_on_epoch(self) -> None:
        self.step.current.reset()
        self.zero_grad.current.reset()

    def load_state_dict(self, state_dict: dict) -> None:
        self.step.load_state_dict(state_dict["step"])
        self.zero_grad.load_state_dict(state_dict["zero_grad"])


@dataclass
class OptimizationProgress(BaseProgress):
    """
    Track optimization progress.

    Args:
        optimizer: Tracks optimizer progress.
        scheduler: Tracks scheduler progress.
    """

    # TODO: support for multiple optimizers
    optimizer: OptimizerProgress = field(default_factory=OptimizerProgress)
    scheduler: Progress = field(default_factory=lambda: Progress.from_defaults(started=None, processed=None))

    @property
    def optimizer_steps(self) -> int:
        return self.optimizer.step.total.completed

    @property
    def scheduler_steps(self) -> int:
        return self.scheduler.total.completed

    def reset_on_epoch(self) -> None:
        self.optimizer.reset_on_epoch()
        self.scheduler.current.reset()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


@dataclass
class EpochLoopProgress(BaseProgress):
    """
    Tracks epoch loop progress.
    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        epoch: Tracks epochs progress.
    """

    epoch: EpochProgress = field(default_factory=EpochProgress)

    def increment_epoch_completed(self) -> None:
        self.epoch.increment_completed()
        self.reset_on_epoch()

    def reset_on_epoch(self) -> None:
        self.epoch.reset_on_epoch()
        self.epoch.current.reset()

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch.load_state_dict(state_dict["epoch"])


@dataclass
class TrainingEpochProgress(EpochProgress):
    """
    Extends ``EpochProgress`` with training specific attributes

    Args:
        total: Tracks the total epoch progress.
        current: Tracks the current epoch progress.
        batch: Tracks batch progress.
        optim: Tracks optimization progress.
        val: Tracks val_loop progress.
    """

    optim: OptimizationProgress = field(default_factory=OptimizationProgress)
    val: EpochLoopProgress = field(default_factory=EpochLoopProgress)

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.optim.load_state_dict(state_dict["optim"])
        self.val.load_state_dict(state_dict["val"])


@dataclass
class FitLoopProgress(EpochLoopProgress):
    """
    Extends ``EpochLoopProgress`` with fit specific attributes

    Args:
        epoch: Tracks epochs progress.
    """

    epoch: TrainingEpochProgress = field(default_factory=TrainingEpochProgress)

    def reset_on_epoch(self) -> None:
        # do not reset `epoch.current` as it should track the number of epochs this `fit` call
        self.epoch.reset_on_epoch()
        self.epoch.optim.reset_on_epoch()
