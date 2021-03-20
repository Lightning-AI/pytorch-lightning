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


def test_invalid_progress_increment(tmpdir):
    """ Make sure that a ValueError is raised for a negative increment. """
    prog = TrainLoopProgress()
    with pytest.raises(ValueError, match=r'.*Increment must be a non-negative'):
        prog.bump_batch(-4)
    with pytest.raises(ValueError, match=r'.*Increment must be a non-negative'):
        prog.bump_epoch(-1)
    with pytest.raises(ValueError, match=r'.*Increment must be a non-negative'):
        prog.bump_step(-100)


def test_bump_batch(tmpdir):
    """ Test sequences for bumping batches. """
    prog = LoopProgress()
    prog.bump_batch(100)
    prog.bump_batch()
    prog.bump_batch(0)
    assert prog.total_batches_processed == 101
    assert prog.batches_processed_this_epoch == 101


def test_bump_epoch(tmpdir):
    """ Test sequences for bumping epochs. """
    prog = LoopProgress()
    prog.bump_epoch(0)
    prog.bump_epoch(4)
    prog.bump_epoch()
    assert prog.total_epochs_processed == 5


def test_train_bump_step(tmpdir):
    """ Test sequences for bumping steps. """
    prog = TrainLoopProgress()
    prog.bump_step(0)
    prog.bump_step(4)
    prog.bump_step()
    assert prog.total_optimizer_steps_processed == 5


def test_reset_batch(tmpdir):
    """ Test sequences for resetting batches. """
    prog = TrainLoopProgress()
    prog.bump_batch(4)
    assert prog.total_batches_processed == 4
    assert prog.batches_processed_this_epoch == 4
    prog.reset_batch_in_epoch()
    assert prog.total_batches_processed == 4
    assert prog.batches_processed_this_epoch == 0
    prog.bump_batch()
    assert prog.total_batches_processed == 5
    assert prog.batches_processed_this_epoch == 1


def test_reset_steps(tmpdir):
    """ Test sequences for resetting steps. """
    prog = TrainLoopProgress()
    prog.bump_step(4)
    assert prog.total_optimizer_steps_processed == 4
    assert prog.optimizer_steps_processed_this_epoch == 4
    prog.reset_step_in_epoch()
    assert prog.total_optimizer_steps_processed == 4
    assert prog.optimizer_steps_processed_this_epoch == 0
    prog.bump_step()
    assert prog.total_optimizer_steps_processed == 5
    assert prog.optimizer_steps_processed_this_epoch == 1
