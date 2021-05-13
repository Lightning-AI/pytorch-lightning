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

import logging

from pytorch_lightning.trainer.progress import LoopProgress


def test_increment_batch_ready(tmpdir):
    prog = LoopProgress()
    prog.increment_batch_ready()
    assert prog.total.ready == 1
    assert prog.epoch.ready == prog.total.ready


def test_increment_batch_started(tmpdir):
    prog = LoopProgress()
    prog.increment_batch_started()
    logging.error(prog)
    assert prog.total.ready == 0
    assert prog.epoch.started == 1
    assert prog.epoch.started == prog.total.started


def test_increment_batch_processed(tmpdir):
    prog = LoopProgress()
    prog.increment_batch_processed()
    assert prog.total.started == 0
    assert prog.total.processed == 1
    assert prog.epoch.processed == prog.total.processed


def test_increment_batch_completed(tmpdir):
    prog = LoopProgress()
    prog.increment_batch_completed()
    assert prog.total.processed == 0
    assert prog.total.completed == 1
    assert prog.epoch.completed == prog.total.completed


def test_increment_epoch(tmpdir):
    """ Test sequences for incrementing epochs. """
    prog = LoopProgress()
    prog.increment_epoch_completed()
    assert prog.epochs_completed_total == 1


def test_increment_batch_ready_start_process_finish_epoch(tmpdir):
    """ Test sequences for incrementing batches reads and epochs. """
    prog = LoopProgress()

    prog.increment_batch_ready()
    assert prog.total.ready == 1
    assert prog.epoch.ready == 1

    prog.increment_batch_started()
    assert prog.total.ready == 1
    assert prog.epoch.ready == 1
    assert prog.total.started == 1
    assert prog.epoch.started == 1

    prog.increment_batch_processed()
    assert prog.total.ready == 1
    assert prog.epoch.ready == 1
    assert prog.total.started == 1
    assert prog.epoch.started == 1
    assert prog.total.processed == 1
    assert prog.epoch.processed == 1

    prog.increment_batch_completed()
    assert prog.total.ready == 1
    assert prog.epoch.ready == 1
    assert prog.total.started == 1
    assert prog.epoch.started == 1
    assert prog.total.processed == 1
    assert prog.epoch.processed == 1
    assert prog.total.completed == 1
    assert prog.epoch.completed == 1

    prog.increment_epoch_completed()
    assert prog.total.ready == 1
    assert prog.epoch.ready == 0
    assert prog.total.started == 1
    assert prog.epoch.started == 0
    assert prog.total.processed == 1
    assert prog.epoch.processed == 0
    assert prog.total.completed == 1
    assert prog.epoch.completed == 0
    assert prog.epochs_completed_total == 1

    prog.increment_batch_ready()
    assert prog.total.ready == 2
    assert prog.epoch.ready == 1
    assert prog.epochs_completed_total == 1

    prog.reset_on_epoch()
    assert prog.epoch.ready == 0


def test_reset_on_epoch(tmpdir):
    """ Test sequences for resetting. """
    prog = LoopProgress()
    prog.increment_batch_ready()
    prog.reset_on_epoch()

    assert prog.total.ready == 1
    assert prog.epoch.ready == 0

    prog.increment_batch_ready()
    assert prog.total.ready == 2
    assert prog.epoch.ready == 1

    prog.increment_epoch_completed()
    assert prog.total.ready == 2
    assert prog.epoch.ready == 0
    assert prog.epochs_completed_total == 1
