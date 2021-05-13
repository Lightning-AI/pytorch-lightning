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
    """ Basic dataclass to track event progress.

    Args:
        ready: Intended to track the number of batches fetched from the dataloader.
        started: Intended to be incremented after `on_*_batch_start` completes.
        processed: Intended to be incremented after `*_step` runs.
        completed: Intended to be incremented after `on_*_batch_end` runs.
    """
    ready: int = 0
    started: int = 0
    processed: int = 0
    completed: int = 0


@dataclass
class LoopProgress:
    """Basic dataclass to track loop progress during execution.

    The general structure is to count totals across epochs as well as intra-epoch counts.
    These counters are local to a trainer rank.
    By default, they are not globally synced across all ranks.

    Args:
        total: Tracks the progress counters across epochs
        epoch: Tracks the progress within the current epoch.
        epochs_completed_total: Track the total number of passes through the dataset
    """
    total = Progress()
    epoch = Progress()
    epochs_completed_total: int = 0

    def increment_batch_ready(self) -> None:
        self.total.ready += 1
        self.epoch.ready += 1

    def increment_batch_started(self) -> None:
        self.total.started += 1
        self.epoch.started += 1

    def increment_batch_processed(self) -> None:
        self.total.processed += 1
        self.epoch.processed += 1

    def increment_batch_completed(self) -> None:
        self.total.completed += 1
        self.epoch.completed += 1

    def increment_epoch_completed(self) -> None:
        self.epochs_completed_total += 1
        self.reset_on_epoch()

    def reset_on_epoch(self) -> None:
        self.epoch = Progress()


@dataclass
class TrainerProgress:
    """ Basic dataclass to track loop progress across stages during trainer execution. """

    train: LoopProgress
    val: LoopProgress
    test: LoopProgress
    predict: LoopProgress
