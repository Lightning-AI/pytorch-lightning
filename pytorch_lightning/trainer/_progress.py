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
    These counters are local to a trainer rank (i.e. they are not by default globally synced across all ranks).

    Args:
        total_epochs_finished: Number of passes through the dataset as defined by the loop
        total_batches_read: Number of batches loaded through the dataloader.
        total_batches_started: Number of batches started processing. This is intended to be incremented after on_train_batch_start runs.
        total_batches_processed: Number of batches processed. This is intended to be incremented after `training_step` runs.
        total_batches_finished: Number of batches finished. This is intended to be incremented after `on_train_batch_end` runs.
        batches_processed_this_epoch: This is reset at the end of the epoch
    """
    total_epochs_finished: int = 0

    total_batches_read: int = 0
    total_batches_started: int = 0
    total_batches_processed: int = 0
    total_batches_finished: int = 0

    batches_read_this_epoch: int = 0
    batches_started_this_epoch: int = 0
    batches_processed_this_epoch: int = 0
    batches_finished_this_epoch: int = 0

    def increment_batch_read(self) -> None:
        self.total_batches_read += 1
        self.batches_read_this_epoch += 1

    def increment_batch_started(self) -> None:
        self.total_batches_started += 1
        self.batches_started_this_epoch += 1

    def increment_batch_processed(self) -> None:
        self.total_batches_processed += 1
        self.batches_processed_this_epoch += 1

    def increment_batch_finished(self) -> None:
        self.total_batches_finished += 1
        self.batches_finished_this_epoch += 1

    def increment_epoch_finished(self) -> None:
        self.total_epochs_finished += 1
        self.reset_on_epoch()

    def reset_on_epoch(self) -> None:
        self.batches_read_this_epoch = 0
        self.batches_started_this_epoch = 0
        self.batches_processed_this_epoch = 0
        self.batches_finished_this_epoch = 0


@dataclass
class TrainLoopProgress(LoopProgress):
    """Extension of ``LoopProgress`` for training specific fields.

    Optimizer steps (parameter updates) are unique to training.
    """

    total_optimizer_steps_processed: int = 0
    optimizer_steps_processed_this_epoch: int = 0

    def increment_optimizer_step(self) -> None:
        self.total_optimizer_steps_processed += 1
        self.optimizer_steps_processed_this_epoch += 1

    def reset_on_epoch(self) -> None:
        super().reset_on_epoch()
        self.optimizer_steps_processed_this_epoch = 0

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
