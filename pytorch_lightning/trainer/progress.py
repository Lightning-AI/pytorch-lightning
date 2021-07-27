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

    def __repr__(self) -> str:
        # hide `None` fields
        args = [f"{k}={v}" for k, v in self.__dict__.items() if v is not None]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def reset_on_restart(self) -> None:
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
        self.total.ready += 1
        self.current.ready += 1

    def increment_started(self) -> None:
        self.total.started += 1
        self.current.started += 1

    def increment_processed(self) -> None:
        self.total.processed += 1
        self.current.processed += 1

    def increment_completed(self) -> None:
        self.total.completed += 1
        self.current.completed += 1

    @classmethod
    def from_defaults(cls, **kwargs: Optional[int]) -> "Progress":
        return cls(total=Tracker(**kwargs), current=Tracker(**kwargs))

    def load_state_dict(self, state_dict: dict) -> None:
        self.total.load_state_dict(state_dict["total"])
        self.current.load_state_dict(state_dict["current"])


@dataclass
class DataLoaderProgress(Progress):
    """
    Tracks the dataloader progress
    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total dataloader progress
        current: Tracks the current dataloader progress
    """

    total: Tracker = field(default_factory=lambda: Tracker(started=None, processed=None))
    current: Tracker = field(default_factory=lambda: Tracker(started=None, processed=None))


@dataclass
class SchedulerProgress(Progress):
    """
    Tracks the scheduler progress
    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total scheduler progress
        current: Tracks the current scheduler progress
    """

    total: Tracker = field(default_factory=lambda: Tracker(started=None, processed=None))
    current: Tracker = field(default_factory=lambda: Tracker(started=None, processed=None))


@dataclass
class OptimizerProgress(BaseProgress):
    """
    Track optimizer progress.

    Args:
        step: Tracks ``optimizer.step`` calls.
        zero_grad: Tracks ``optimizer.zero_grad`` calls.
    """

    step: Progress = field(default_factory=lambda: Progress.from_defaults(started=None, processed=None))
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
        optimizer_idx: The index of the current optimizer.
    """

    # TODO: support for multiple optimizers
    optimizer: OptimizerProgress = field(default_factory=OptimizerProgress)
    optimizer_idx: int = 0

    @property
    def optimizer_steps(self) -> int:
        return self.optimizer.step.total.completed

    def reset_on_epoch(self) -> None:
        self.optimizer.reset_on_epoch()
        self.optimizer_idx = 0

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.optimizer_idx = state_dict["optimizer_idx"]
