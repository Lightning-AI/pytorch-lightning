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

from dataclasses import dataclass, field


@dataclass
class ProgressState:
    """
    Basic dataclass to track event progress.

    Args:
        ready: Intended to track the number of events ready to start.
        started: Intended to be incremented after the event is started (e.g. after `on_*_start runs).
        processed: Intended to be incremented after the event is processed.
        completed: Intended to be incremented after the event completes (e.g. after `on_*_end` runs).
    """
    ready: int = 0
    started: int = 0
    processed: int = 0
    completed: int = 0

    def reset(self) -> None:
        self.ready = 0
        self.started = 0
        self.processed = 0
        self.completed = 0


@dataclass
class Progress:
    """
    Basic dataclass to track aggregated and current progress states.

    Args:
        total: Intended to track the total progress of an event
        current: Intended to track the current progress of an event
    """
    total: ProgressState = field(default_factory=ProgressState)
    current: ProgressState = field(default_factory=ProgressState)

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


@dataclass
class LoopProgress:
    """
    Dataclass to track loop progress during execution.

    These counters are local to a trainer rank. By default, they are not globally synced across all ranks.
    Args:
        epoch: Tracks epochs progress.
        batch: Tracks batch progress.
    """
    epoch: Progress = field(default_factory=Progress)
    batch: Progress = field(default_factory=Progress)

    def increment_epoch_completed(self) -> None:
        self.epoch.increment_completed()
        self.reset_on_epoch()

    def reset_on_epoch(self) -> None:
        self.batch.current.reset()
        self.epoch.current.reset()
