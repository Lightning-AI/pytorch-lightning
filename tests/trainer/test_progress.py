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

import pytest

from pytorch_lightning.trainer.progress import LoopProgress, TrainLoopProgress


def test_increment_batch(tmpdir):
    """ Test sequences for incrementing batches. """
    prog = LoopProgress()
    prog.increment_batch()
    prog.increment_batch()
    assert prog.total_batches_processed == 2
    assert prog.batches_processed_this_epoch == 2


def test_train_increment_step(tmpdir):
    """ Test sequences for bumping steps. """
    prog = TrainLoopProgress()
    prog.increment_step()
    assert prog.total_optimizer_steps_processed == 1


def test_increment_epoch(tmpdir):
    """ Test sequences for incrementing epochs. """
    prog = LoopProgress()
    prog.increment_epoch()
    prog.increment_epoch()
    assert prog.total_epochs_processed == 2


def test_increment_batch_epoch(tmpdir):
    """ Test sequences for incrementing epochs. """
    prog = LoopProgress()
    prog.increment_epoch()
    prog.increment_epoch()
    assert prog.total_epochs_processed == 2


def test_increment_batch_epoch(tmpdir):
    """ Test sequences for incrementing batches and epochs. """
    prog = LoopProgress()
    prog.increment_batch()
    assert prog.total_batches_processed == 1
    assert prog.batches_processed_this_epoch == 1
    prog.increment_epoch()
    assert prog.total_batches_processed == 1
    assert prog.batches_processed_this_epoch == 0
    assert prog.total_epochs_processed == 1
    prog.increment_batch()
    assert prog.total_batches_processed == 2
    assert prog.batches_processed_this_epoch == 1
    assert prog.total_epochs_processed == 1


def test_increment_batch_step_epoch(tmpdir):
    """ Test sequences for incrementing batches, steps, and epochs. """
    prog = TrainLoopProgress()
    prog.increment_batch()
    assert prog.total_batches_processed == 1
    assert prog.batches_processed_this_epoch == 1
    prog.increment_step()
    assert prog.total_optimizer_steps_processed == 1
    assert prog.optimizer_steps_processed_this_epoch == 1
    prog.increment_epoch()
    assert prog.total_batches_processed == 1
    assert prog.batches_processed_this_epoch == 0
    assert prog.total_epochs_processed == 1
    assert prog.total_optimizer_steps_processed == 1
    assert prog.optimizer_steps_processed_this_epoch == 0
    prog.increment_batch()
    assert prog.total_batches_processed == 2
    assert prog.batches_processed_this_epoch == 1
    assert prog.total_epochs_processed == 1
    prog.increment_step()
    assert prog.total_optimizer_steps_processed == 2
    assert prog.optimizer_steps_processed_this_epoch == 1
    assert prog.total_epochs_processed == 1


def test_reset_batch(tmpdir):
    """ Test sequences for resetting batch counts. """
    prog = TrainLoopProgress()
    prog.increment_batch()
    assert prog.total_batches_processed == 1
    assert prog.batches_processed_this_epoch == 1
    prog.reset_batch_in_epoch()
    assert prog.total_batches_processed == 1
    assert prog.batches_processed_this_epoch == 0
    prog.increment_batch()
    assert prog.total_batches_processed == 2
    assert prog.batches_processed_this_epoch == 1


def test_reset_steps(tmpdir):
    """ Test sequences for resetting steps. """
    prog = TrainLoopProgress()
    prog.increment_step()
    assert prog.total_optimizer_steps_processed == 1
    assert prog.optimizer_steps_processed_this_epoch == 1
    prog.reset_step_in_epoch()
    assert prog.total_optimizer_steps_processed == 1
    assert prog.optimizer_steps_processed_this_epoch == 0
    prog.increment_step()
    assert prog.total_optimizer_steps_processed == 2
    assert prog.optimizer_steps_processed_this_epoch == 1
