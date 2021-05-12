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
class LoopProgress:
    """Basic dataclass to track loop progress during execution.

    The general structure is to count totals across epochs as well as intra-epoch counts.
    These counters are local to a trainer rank.
    By default, they are not by globally synced across all ranks.

    Args:
        epochs_finished_total: Number of passes through the dataset.
            This is intended to be incremented after `on_*_epoch_end` completes.
        batches_read_total: Number of batches loaded through the dataloader.
        batches_started_total: Number of batches started processing.
            This is intended to be incremented after `on_*_batch_start` completes.
        batches_processed_total: Number of batches processed.
            This is intended to be incremented after `*_step` runs.
        batches_finished_total: Number of batches finished.
            This is intended to be incremented after `on_*_batch_end` runs.

        batches_read_epoch: Number of batches loaded through the dataloader within the current epoch.
        batches_started_epoch: Number of batches started processing within the current epoch.
            This is intended to be incremented after `on_*_batch_start` completes.
        batches_processed_epoch: Number of batches processed within the current epoch.
            This is intended to be incremented after `*_step` runs.
        batches_finished_epoch: Number of batches finished within the current epoch.
            This is intended to be incremented after `on_*_batch_end` runs.
    """
    epochs_finished_total: int = 0

    batches_read_total: int = 0
    batches_started_total: int = 0
    batches_processed_total: int = 0
    batches_finished_total: int = 0

    batches_read_epoch: int = 0
    batches_started_epoch: int = 0
    batches_processed_epoch: int = 0
    batches_finished_epoch: int = 0

    def increment_batch_read(self) -> None:
        self.batches_read_total += 1
        self.batches_read_epoch += 1

    def increment_batch_started(self) -> None:
        self.batches_started_total += 1
        self.batches_started_epoch += 1

    def increment_batch_processed(self) -> None:
        self.batches_processed_total += 1
        self.batches_processed_epoch += 1

    def increment_batch_finished(self) -> None:
        self.batches_finished_total += 1
        self.batches_finished_epoch += 1

    def increment_epoch_finished(self) -> None:
        self.epochs_finished_total += 1
        self.reset_on_epoch()

    def reset_on_epoch(self) -> None:
        self.batches_read_epoch = 0
        self.batches_started_epoch = 0
        self.batches_processed_epoch = 0
        self.batches_finished_epoch = 0


@dataclass
class TrainLoopProgress(LoopProgress):
    """Extension of ``LoopProgress`` for training specific fields.

    Optimizer steps (parameter updates) are unique to training.
    """

    optimizer_steps_processed_total: int = 0
    optimizer_steps_processed_epoch: int = 0

    def increment_optimizer_step(self) -> None:
        self.optimizer_steps_processed_total += 1
        self.optimizer_steps_processed_epoch += 1

    def reset_on_epoch(self) -> None:
        super().reset_on_epoch()
        self.optimizer_steps_processed_epoch = 0

    def increment_epoch_finished(self) -> None:
        super().increment_epoch_finished()
        self.reset_on_epoch()


@dataclass
class Progress:
    """ Basic dataclass to track loop progress across stages during trainer execution. """

    train_progress: TrainLoopProgress
    val_progress: LoopProgress
    test_progress: LoopProgress
    predict_progress: LoopProgress
