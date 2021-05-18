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

from pytorch_lightning.trainer.progress import LoopProgress, ProgressState


def test_increment_ready(tmpdir):
    prog = LoopProgress()
    prog.batch.increment_ready()
    assert prog.batch.total.ready == 1
    assert prog.batch.current.ready == prog.batch.total.ready


def test_increment_started(tmpdir):
    prog = LoopProgress()
    prog.epoch.increment_started()
    assert prog.batch.total.ready == 0
    assert prog.epoch.total.ready == 0
    assert prog.epoch.total.started == 1
    assert prog.epoch.total.started == prog.epoch.current.started


def test_increment_processed(tmpdir):
    prog = LoopProgress()
    prog.epoch.increment_processed()
    assert prog.batch.total.ready == 0
    assert prog.batch.total.started == 0
    assert prog.epoch.total.started == 0
    assert prog.epoch.total.processed == 1
    assert prog.epoch.total.processed == prog.epoch.current.processed


def test_increment_completed(tmpdir):
    prog = LoopProgress()
    prog.epoch.increment_completed()
    assert prog.batch.total.ready == 0
    assert prog.batch.total.started == 0
    assert prog.epoch.total.started == 0
    assert prog.epoch.total.processed == 0
    assert prog.epoch.total.completed == 1
    assert prog.epoch.total.completed == prog.epoch.current.completed


def test_increment_epoch(tmpdir):
    """ Test sequences for incrementing epochs. """
    prog = LoopProgress()
    prog.batch.increment_completed()
    assert prog.batch.current.completed == 1

    prog.increment_epoch_completed()
    prog.increment_epoch_completed()
    assert prog.epoch.current.completed == 0
    assert prog.epoch.total.completed == 2
    assert prog.batch.current.completed == 0
    assert prog.batch.total.completed == 1


def test_reset_on_epoch(tmpdir):
    """ Test sequences for resetting. """
    prog = LoopProgress()

    prog.batch.increment_started()
    assert prog.batch.total.started == 1
    assert prog.epoch.total.started == 0

    prog.reset_on_epoch()
    assert prog.batch.current.started == 0
    assert prog.batch.total == ProgressState(started=1)

    prog.batch.increment_started()
    assert prog.batch.total == ProgressState(started=2)
    assert prog.epoch.total.started == 0


def test_increment_batch_ready_start_process_finish_epoch(tmpdir):
    """ Test sequences for incrementing batches reads and epochs. """
    prog = LoopProgress()

    prog.epoch.increment_ready()
    assert prog.epoch.total.ready == 1
    assert prog.epoch.current.ready == 1
    assert prog.batch.total.ready == 0
    assert prog.batch.current.ready == 0

    prog.epoch.increment_started()
    assert prog.epoch.total.ready == 1
    assert prog.epoch.current.ready == 1
    assert prog.epoch.total.started == 1
    assert prog.epoch.current.started == 1
    assert prog.batch.total.started == 0
    assert prog.batch.current.started == 0

    prog.batch.increment_ready()
    assert prog.batch.total.ready == 1
    assert prog.batch.current.ready == 1
    assert prog.epoch.total.ready == 1
    assert prog.epoch.current.ready == 1

    prog.batch.increment_started()
    assert prog.batch.total.started == 1
    assert prog.batch.current.started == 1
    assert prog.epoch.total.started == 1
    assert prog.epoch.current.started == 1

    prog.batch.increment_processed()
    assert prog.batch.total.processed == 1
    assert prog.batch.current.processed == 1
    assert prog.epoch.total.processed == 0
    assert prog.epoch.current.processed == 0

    prog.batch.increment_completed()
    assert prog.batch.total.completed == 1
    assert prog.batch.current.completed == 1
    assert prog.epoch.total.completed == 0
    assert prog.epoch.current.completed == 0

    prog.epoch.increment_processed()
    assert prog.batch.total.processed == 1
    assert prog.batch.current.processed == 1
    assert prog.epoch.total.processed == 1
    assert prog.epoch.current.processed == 1

    prog.increment_epoch_completed()
    assert prog.batch.total.completed == 1
    assert prog.batch.current.completed == 0
    assert prog.epoch.total.completed == 1
    assert prog.epoch.current.completed == 0

    prog.epoch.increment_ready()
    assert prog.epoch.total.ready == 2
    assert prog.epoch.current.ready == 1

    prog.batch.increment_ready()
    assert prog.batch.total.ready == 2
    assert prog.batch.current.ready == 1

    prog.reset_on_epoch()
    assert prog.batch.current.ready == 0
    assert prog.epoch.current.ready == 0
