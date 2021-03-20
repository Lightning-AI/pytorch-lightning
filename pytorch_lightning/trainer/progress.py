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

    Args:
        total_epochs_processed: Number of passes through the dataset as defined by the loop
        total_batches_processed: Number of batches seen. This monotonically increases throughout execution
        batches_processed_this_epoch: This is reset at the end of the epoch
    """
    total_epochs_processed: int = 0
    total_batches_processed: int = 0
    batches_processed_this_epoch: int = 0

    def bump_batch(self, increment: int = 1) -> None:
        if increment < 0:
            raise ValueError(f"Increment must be a non-negative value but received {increment}")
        self.total_batches_processed += increment
        self.batches_processed_this_epoch += increment

    def bump_epoch(self, increment: int = 1) -> None:
        if increment < 0:
            raise ValueError(f"Increment must be a non-negative value but received {increment}")
        self.total_epochs_processed += increment

    def reset_batch_in_epoch(self) -> None:
        self.batches_processed_this_epoch = 0


@dataclass
class TrainLoopProgress(LoopProgress):
    """Extension of ``LoopProgress`` for training specific fields.

    Optimizer steps (parameter updates) are unique to training.
    """

    total_optimizer_steps_processed: int = 0
    optimizer_steps_processed_this_epoch: int = 0

    def bump_step(self, increment: int = 1) -> None:
        if increment < 0:
            raise ValueError(f"Increment must be a non-negative value but received {increment}")
        self.total_optimizer_steps_processed += increment
        self.optimizer_steps_processed_this_epoch += increment

    def reset_step_in_epoch(self) -> None:
        self.optimizer_steps_processed_this_epoch = 0


@dataclass
class Progress:
    """ Basic dataclass to track loop progress across stages during trainer execution. """

    train_progress: TrainLoopProgress
    val_progress: LoopProgress
    test_progress: LoopProgress
    predict_progress: LoopProgress
