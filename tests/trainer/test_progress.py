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

from pytorch_lightning.trainer.progress import LoopProgress, TrainLoopProgress


def test_increment_batch_read(tmpdir):
    prog = LoopProgress()
    prog.increment_batch_read()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == prog.batches_read_total


def test_increment_batch_started(tmpdir):
    prog = LoopProgress()
    prog.increment_batch_started()
    assert prog.batches_read_total == 0
    assert prog.batches_started_epoch == 1
    assert prog.batches_started_epoch == prog.batches_started_total


def test_increment_batch_processed(tmpdir):
    prog = LoopProgress()
    prog.increment_batch_processed()
    assert prog.batches_started_total == 0
    assert prog.batches_processed_total == 1
    assert prog.batches_processed_epoch == prog.batches_processed_total


def test_increment_batch_finished(tmpdir):
    prog = LoopProgress()
    prog.increment_batch_finished()
    assert prog.batches_processed_total == 0
    assert prog.batches_finished_total == 1
    assert prog.batches_finished_epoch == prog.batches_finished_total


def test_train_increment_step(tmpdir):
    """ Test sequences for incrementing optimizer steps. """
    prog = TrainLoopProgress()
    prog.increment_optimizer_step()
    assert prog.optimizer_steps_processed_total == 1
    assert prog.optimizer_steps_processed_total == prog.optimizer_steps_processed_epoch


def test_increment_epoch(tmpdir):
    """ Test sequences for incrementing epochs. """
    prog = LoopProgress()
    prog.increment_epoch_finished()
    assert prog.epochs_finished_total == 1


def test_increment_batch_read_start_process_finish_epoch(tmpdir):
    """ Test sequences for incrementing batches reads and epochs. """
    prog = LoopProgress()

    prog.increment_batch_read()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 1

    prog.increment_batch_started()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 1
    assert prog.batches_started_total == 1
    assert prog.batches_started_epoch == 1

    prog.increment_batch_processed()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 1
    assert prog.batches_started_total == 1
    assert prog.batches_started_epoch == 1
    assert prog.batches_processed_total == 1
    assert prog.batches_processed_epoch == 1

    prog.increment_batch_finished()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 1
    assert prog.batches_started_total == 1
    assert prog.batches_started_epoch == 1
    assert prog.batches_processed_total == 1
    assert prog.batches_processed_epoch == 1
    assert prog.batches_finished_total == 1
    assert prog.batches_finished_epoch == 1

    prog.increment_epoch_finished()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 0
    assert prog.batches_started_total == 1
    assert prog.batches_started_epoch == 0
    assert prog.batches_processed_total == 1
    assert prog.batches_processed_epoch == 0
    assert prog.batches_finished_total == 1
    assert prog.batches_finished_epoch == 0
    assert prog.epochs_finished_total == 1

    prog.increment_batch_read()
    assert prog.batches_read_total == 2
    assert prog.batches_read_epoch == 1
    assert prog.epochs_finished_total == 1

    prog.reset_on_epoch()
    assert prog.batches_read_epoch == 0


def test_increment_batch_read_start_process_step_finish_epoch(tmpdir):
    """ Test sequences for incrementing batches reads and epochs. """
    prog = TrainLoopProgress()

    prog.increment_batch_read()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 1

    prog.increment_batch_started()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 1
    assert prog.batches_started_total == 1
    assert prog.batches_started_epoch == 1

    prog.increment_batch_processed()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 1
    assert prog.batches_started_total == 1
    assert prog.batches_started_epoch == 1
    assert prog.batches_processed_total == 1
    assert prog.batches_processed_epoch == 1

    prog.increment_optimizer_step()
    assert prog.optimizer_steps_processed_total == 1
    assert prog.optimizer_steps_processed_epoch == 1

    prog.increment_batch_finished()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 1
    assert prog.batches_started_total == 1
    assert prog.batches_started_epoch == 1
    assert prog.batches_processed_total == 1
    assert prog.batches_processed_epoch == 1
    assert prog.batches_finished_total == 1
    assert prog.batches_finished_epoch == 1

    prog.increment_epoch_finished()
    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 0
    assert prog.batches_started_total == 1
    assert prog.batches_started_epoch == 0
    assert prog.batches_processed_total == 1
    assert prog.batches_processed_epoch == 0
    assert prog.optimizer_steps_processed_total == 1
    assert prog.optimizer_steps_processed_epoch == 0
    assert prog.batches_finished_total == 1
    assert prog.batches_finished_epoch == 0
    assert prog.epochs_finished_total == 1

    prog.increment_batch_read()
    assert prog.batches_read_total == 2
    assert prog.batches_read_epoch == 1
    assert prog.epochs_finished_total == 1

    prog.reset_on_epoch()
    assert prog.optimizer_steps_processed_epoch == 0


def test_reset_on_epoch(tmpdir):
    """ Test sequences for resetting. """
    prog = TrainLoopProgress()
    prog.increment_batch_read()
    prog.increment_optimizer_step()
    prog.reset_on_epoch()

    assert prog.batches_read_total == 1
    assert prog.batches_read_epoch == 0
    assert prog.optimizer_steps_processed_total == 1
    assert prog.optimizer_steps_processed_epoch == 0

    prog.increment_batch_read()
    assert prog.batches_read_total == 2
    assert prog.batches_read_epoch == 1
    assert prog.optimizer_steps_processed_total == 1
    assert prog.optimizer_steps_processed_epoch == 0

    prog.increment_optimizer_step()
    assert prog.optimizer_steps_processed_total == 2
    assert prog.optimizer_steps_processed_epoch == 1
