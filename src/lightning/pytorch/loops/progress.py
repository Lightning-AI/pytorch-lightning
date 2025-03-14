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

from typing_extensions import override


@dataclass
class _BaseProgress:
    """Mixin that implements state-loading utilities for dataclasses."""

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "_BaseProgress":
        obj = cls()
        obj.load_state_dict(state_dict)
        return obj

    def reset(self) -> None:
        """Reset the object's state."""
        raise NotImplementedError


@dataclass
class _ReadyCompletedTracker(_BaseProgress):
    """Track an event's progress.

    Args:
        ready: Intended to track the number of events ready to start.
        completed: Intended to be incremented after the event completes (e.g. after ``on_*_end`` runs).

    These attributes should be increased in order, that is, :attr:`ready` first and :attr:`completed` last.

    """

    ready: int = 0
    completed: int = 0

    @override
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

    def increment_by(self, n: int) -> None:
        self.ready += n
        self.completed += n


@dataclass
class _StartedTracker(_ReadyCompletedTracker):
    """Track an event's progress.

    Args:
        ready: Intended to track the number of events ready to start.
        started: Intended to be incremented after the event is started (e.g. after ``on_*_start`` runs).
        completed: Intended to be incremented after the event completes (e.g. after ``on_*_end`` runs).

    These attributes should be increased in order, that is, :attr:`ready` first and :attr:`completed` last.

    """

    started: int = 0

    @override
    def reset(self) -> None:
        super().reset()
        self.started = 0

    @override
    def reset_on_restart(self) -> None:
        super().reset_on_restart()
        self.started = self.completed

    @override
    def increment_by(self, n: int) -> None:
        super().increment_by(n)
        self.started += n


@dataclass
class _ProcessedTracker(_StartedTracker):
    """Track an event's progress.

    Args:
        ready: Intended to track the number of events ready to start.
        started: Intended to be incremented after the event is started (e.g. after ``on_*_start`` runs).
        processed: Intended to be incremented after the event is processed.
        completed: Intended to be incremented after the event completes (e.g. after ``on_*_end`` runs).

    These attributes should be increased in order, that is, :attr:`ready` first and :attr:`completed` last.

    """

    processed: int = 0

    @override
    def reset(self) -> None:
        super().reset()
        self.processed = 0

    @override
    def reset_on_restart(self) -> None:
        super().reset_on_restart()
        self.processed = self.completed

    @override
    def increment_by(self, n: int) -> None:
        super().increment_by(n)
        self.processed += n


@dataclass
class _Progress(_BaseProgress):
    """Track aggregated and current progress.

    Args:
        total: Intended to track the total progress of an event.
        current: Intended to track the current progress of an event.

    """

    total: _ReadyCompletedTracker = field(default_factory=_ProcessedTracker)
    current: _ReadyCompletedTracker = field(default_factory=_ProcessedTracker)

    def __post_init__(self) -> None:
        if self.total.__class__ is not self.current.__class__:
            raise ValueError("The `total` and `current` instances should be of the same class")

    def increment_ready(self) -> None:
        self.total.ready += 1
        self.current.ready += 1

    def increment_started(self) -> None:
        if not isinstance(self.total, _StartedTracker):
            raise TypeError(f"`{self.total.__class__.__name__}` doesn't have a `started` attribute")
        self.total.started += 1
        self.current.started += 1

    def increment_processed(self) -> None:
        if not isinstance(self.total, _ProcessedTracker):
            raise TypeError(f"`{self.total.__class__.__name__}` doesn't have a `processed` attribute")
        self.total.processed += 1
        self.current.processed += 1

    def increment_completed(self) -> None:
        self.total.completed += 1
        self.current.completed += 1

    @classmethod
    def from_defaults(cls, tracker_cls: type[_ReadyCompletedTracker], **kwargs: int) -> "_Progress":
        """Utility function to easily create an instance from keyword arguments to both ``Tracker``s."""
        return cls(total=tracker_cls(**kwargs), current=tracker_cls(**kwargs))

    @override
    def reset(self) -> None:
        self.total.reset()
        self.current.reset()

    def reset_on_run(self) -> None:
        self.current.reset()

    def reset_on_restart(self) -> None:
        self.current.reset_on_restart()

    def increment_by(self, n: int) -> None:
        self.total.increment_by(n)
        self.current.increment_by(n)

    @override
    def load_state_dict(self, state_dict: dict) -> None:
        self.total.load_state_dict(state_dict["total"])
        self.current.load_state_dict(state_dict["current"])


@dataclass
class _BatchProgress(_Progress):
    """Tracks batch progress.

    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total batch progress.
        current: Tracks the current batch progress.
        is_last_batch: Whether the batch is the last one. This is useful for iterable datasets.

    """

    is_last_batch: bool = False

    @override
    def reset(self) -> None:
        super().reset()
        self.is_last_batch = False

    @override
    def reset_on_run(self) -> None:
        super().reset_on_run()
        self.is_last_batch = False

    def increment_by(self, n: int, is_last_batch: bool = False) -> None:
        super().increment_by(n)
        self.is_last_batch = is_last_batch

    @override
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.is_last_batch = state_dict["is_last_batch"]


@dataclass
class _SchedulerProgress(_Progress):
    """Tracks scheduler progress.

    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the total scheduler progress.
        current: Tracks the current scheduler progress.

    """

    total: _ReadyCompletedTracker = field(default_factory=_ReadyCompletedTracker)
    current: _ReadyCompletedTracker = field(default_factory=_ReadyCompletedTracker)


@dataclass
class _OptimizerProgress(_BaseProgress):
    """Track optimizer progress.

    Args:
        step: Tracks ``optimizer.step`` calls.
        zero_grad: Tracks ``optimizer.zero_grad`` calls.

    """

    step: _Progress = field(default_factory=lambda: _Progress.from_defaults(_ReadyCompletedTracker))
    zero_grad: _Progress = field(default_factory=lambda: _Progress.from_defaults(_StartedTracker))

    @override
    def reset(self) -> None:
        self.step.reset()
        self.zero_grad.reset()

    def reset_on_run(self) -> None:
        self.step.reset_on_run()
        self.zero_grad.reset_on_run()

    def reset_on_restart(self) -> None:
        self.step.reset_on_restart()
        self.zero_grad.reset_on_restart()

    @override
    def load_state_dict(self, state_dict: dict) -> None:
        self.step.load_state_dict(state_dict["step"])
        self.zero_grad.load_state_dict(state_dict["zero_grad"])


@dataclass
class _OptimizationProgress(_BaseProgress):
    """Track optimization progress.

    Args:
        optimizer: Tracks optimizer progress.

    """

    optimizer: _OptimizerProgress = field(default_factory=_OptimizerProgress)

    @property
    def optimizer_steps(self) -> int:
        return self.optimizer.step.total.completed

    @override
    def reset(self) -> None:
        self.optimizer.reset()

    def reset_on_run(self) -> None:
        self.optimizer.reset_on_run()

    def reset_on_restart(self) -> None:
        self.optimizer.reset_on_restart()

    @override
    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
