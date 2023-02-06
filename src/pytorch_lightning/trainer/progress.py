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
from dataclasses import asdict, dataclass, field
from typing import Type


@dataclass
class BaseProgress:
    """Mixin that implements state-loading utilities for dataclasses."""

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "BaseProgress":
        obj = cls()
        obj.load_state_dict(state_dict)
        return obj

    def reset(self) -> None:
        """Reset the object's state."""
        raise NotImplementedError


@dataclass
class ReadyCompletedTracker(BaseProgress):
    """Track an event's progress.

    Args:
        ready: Intended to track the number of events ready to start.
        completed: Intended to be incremented after the event completes (e.g. after ``on_*_end`` runs).

    These attributes should be increased in order, that is, :attr:`ready` first and :attr:`completed` last.
    """

    ready: int = 0
    completed: int = 0

    def reset(self) -> None:
        """Reset the state."""
        self.ready = 0
        self.completed = 0

    def reset_on_restart(self) -> None:
        """Reset the progress on restart.

        If there is a failure before all attributes are increased, restore the attributes to the last fully completed
        value.
        """
        self.ready = self.completed


@dataclass
class StartedTracker(ReadyCompletedTracker):
    """Track an event's progress.

    Args:
        ready: Intended to track the number of events ready to start.
        started: Intended to be incremented after the event is started (e.g. after ``on_*_start`` runs).
        completed: Intended to be incremented after the event completes (e.g. after ``on_*_end`` runs).

    These attributes should be increased in order, that is, :attr:`ready` first and :attr:`completed` last.
    """

    started: int = 0

    def reset(self) -> None:
        super().reset()
        self.started = 0

    def reset_on_restart(self) -> None:
        super().reset_on_restart()
        self.started = self.completed


@dataclass
class ProcessedTracker(StartedTracker):
    """Track an event's progress.

    Args:
        ready: Intended to track the number of events ready to start.
        started: Intended to be incremented after the event is started (e.g. after ``on_*_start`` runs).
        processed: Intended to be incremented after the event is processed.
        completed: Intended to be incremented after the event completes (e.g. after ``on_*_end`` runs).

    These attributes should be increased in order, that is, :attr:`ready` first and :attr:`completed` last.
    """

    processed: int = 0

    def reset(self) -> None:
        super().reset()
        self.processed = 0

    def reset_on_restart(self) -> None:
        super().reset_on_restart()
        self.processed = self.completed


@dataclass
class Progress(BaseProgress):
    """Track aggregated and current progress.

    Args:
        total: Intended to track the total progress of an event.
        current: Intended to track the current progress of an event.
    """

    total: ReadyCompletedTracker = field(default_factory=ProcessedTracker)
    current: ReadyCompletedTracker = field(default_factory=ProcessedTracker)

    def __post_init__(self) -> None:
        if type(self.total) is not type(self.current):  # noqa: E721
            raise ValueError("The `total` and `current` instances should be of the same class")

    def increment_ready(self) -> None:
        self.total.ready += 1
        self.current.ready += 1

    def increment_started(self) -> None:
        if not isinstance(self.total, StartedTracker):
            raise TypeError(f"`{self.total.__class__.__name__}` doesn't have a `started` attribute")
        self.total.started += 1
        self.current.started += 1

    def increment_processed(self) -> None:
        if not isinstance(self.total, ProcessedTracker):
            raise TypeError(f"`{self.total.__class__.__name__}` doesn't have a `processed` attribute")
        self.total.processed += 1
        self.current.processed += 1

    def increment_completed(self) -> None:
        self.total.completed += 1
        self.current.completed += 1

    @classmethod
    def from_defaults(cls, tracker_cls: Type[ReadyCompletedTracker], **kwargs: int) -> "Progress":
        """Utility function to easily create an instance from keyword arguments to both ``Tracker``s."""
        return cls(total=tracker_cls(**kwargs), current=tracker_cls(**kwargs))

    def reset(self) -> None:
        self.total.reset()
        self.current.reset()

    def reset_on_run(self) -> None:
        self.current.reset()

    def reset_on_restart(self) -> None:
        self.current.reset_on_restart()

    def load_state_dict(self, state_dict: dict) -> None:
        self.total.load_state_dict(state_dict["total"])
        self.current.load_state_dict(state_dict["current"])


@dataclass
class DataLoaderProgress(Progress):
    """Tracks dataloader progress.

    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total dataloader progress.
        current: Tracks the current dataloader progress.
    """

    total: ReadyCompletedTracker = field(default_factory=ReadyCompletedTracker)
    current: ReadyCompletedTracker = field(default_factory=ReadyCompletedTracker)


@dataclass
class BatchProgress(Progress):
    """Tracks batch progress.

    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total batch progress.
        current: Tracks the current batch progress.
        is_last_batch: Whether the batch is the last one. This is useful for iterable datasets.
    """

    is_last_batch: bool = False

    def reset(self) -> None:
        super().reset()
        self.is_last_batch = False

    def reset_on_run(self) -> None:
        super().reset_on_run()
        self.is_last_batch = False

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.is_last_batch = state_dict["is_last_batch"]


@dataclass
class SchedulerProgress(Progress):
    """Tracks scheduler progress.

    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total scheduler progress.
        current: Tracks the current scheduler progress.
    """

    total: ReadyCompletedTracker = field(default_factory=ReadyCompletedTracker)
    current: ReadyCompletedTracker = field(default_factory=ReadyCompletedTracker)


@dataclass
class OptimizerProgress(BaseProgress):
    """Track optimizer progress.

    Args:
        step: Tracks ``optimizer.step`` calls.
        zero_grad: Tracks ``optimizer.zero_grad`` calls.
    """

    step: Progress = field(default_factory=lambda: Progress.from_defaults(ReadyCompletedTracker))
    zero_grad: Progress = field(default_factory=lambda: Progress.from_defaults(StartedTracker))

    def reset(self) -> None:
        self.step.reset()
        self.zero_grad.reset()

    def reset_on_run(self) -> None:
        self.step.reset_on_run()
        self.zero_grad.reset_on_run()

    def reset_on_restart(self) -> None:
        self.step.reset_on_restart()
        self.zero_grad.reset_on_restart()

    def load_state_dict(self, state_dict: dict) -> None:
        self.step.load_state_dict(state_dict["step"])
        self.zero_grad.load_state_dict(state_dict["zero_grad"])


@dataclass
class OptimizationProgress(BaseProgress):
    """Track optimization progress.

    Args:
        optimizer: Tracks optimizer progress.
        optimizer_position: The index of the current optimizer amongst the currently active optimizers.
            Used to know which optimizer we were using when restarting.
            Since not all optimizers may be active at a given time, this index is different from the ``optimizer_idx``
            seen in the optimization loops.
    """

    # TODO: support for multiple optimizers
    optimizer: OptimizerProgress = field(default_factory=OptimizerProgress)
    optimizer_position: int = 0

    @property
    def optimizer_steps(self) -> int:
        return self.optimizer.step.total.completed

    def reset(self) -> None:
        self.optimizer.reset()
        self.optimizer_position = 0

    def reset_on_run(self) -> None:
        self.optimizer.reset_on_run()
        self.optimizer_position = 0

    def reset_on_restart(self) -> None:
        self.optimizer.reset_on_restart()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.optimizer_position = state_dict["optimizer_position"]
