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
from dataclasses import dataclass


@dataclass
class Progress:
    """
    Basic dataclass to track event progress.

    Args:
        ready: Intended to track the number of events ready to start.
        started: Intended to be incremented after `on_*_start` completes.
        processed: Intended to be incremented after the event is processed.
        completed: Intended to be incremented after `on_*_end` completes.
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
class BaseProgress:
    """
    Basic dataclass to track event progress.

    Args:
        total: Intended to track the total progress of an event
        current: Intended to track the current progress of an event
    """
    total: Progress = Progress()
    current: Progress = Progress()

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
    epoch: BaseProgress = BaseProgress()
    batch: BaseProgress = BaseProgress()

    def increment_batch_ready(self) -> None:
        self.batch.increment_ready()

    def increment_batch_started(self) -> None:
        self.batch.increment_started()

    def increment_batch_processed(self) -> None:
        self.batch.increment_processed()

    def increment_batch_completed(self) -> None:
        self.batch.increment_completed()

    def increment_epoch_completed(self) -> None:
        self.epoch.total.completed += 1
        self.reset_on_epoch()

    def reset_on_epoch(self) -> None:
        self.epoch.current.reset()


@dataclass
class OptimizationProgress:
    """
    Dataclass to track optimization progress.

    Args:
        optimizer: Tracks optimizer progress.
        scheduler: Tracks scheduler progress.
    """
    optimizer: BaseProgress = BaseProgress()
    scheduler: BaseProgress = BaseProgress()

    @property
    def optimizer_steps(self) -> int:
        return self.optimizer.total.completed

    @property
    def scheduler_steps(self) -> int:
        return self.scheduler.total.completed


@dataclass
class TrainProgress(BaseProgress):
    """
    Extends the progress with training specific attributes

    Args:
        optimization: Tracks optimization progress
    """
    optimization: OptimizationProgress = OptimizationProgress()


@dataclass
class TrainLoopProgress(LoopProgress):
    epoch: TrainProgress = TrainProgress()


@dataclass
class FitLoopProgress:
    train: TrainProgress = TrainLoopProgress()
    val: LoopProgress = LoopProgress()


@dataclass
class LoopState:
    """
    Basic dataclass to track loop progress across trainer functions during trainer execution.

    This class will be removed and these attributes will live in each loop.
    """

    fit: FitLoopProgress = FitLoopProgress()
    val: LoopProgress = LoopProgress()
    test: LoopProgress = LoopProgress()
    predict: LoopProgress = LoopProgress()
