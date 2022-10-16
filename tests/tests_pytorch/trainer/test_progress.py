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
from copy import deepcopy

import pytest

from pytorch_lightning.trainer.progress import (
    BaseProgress,
    OptimizerProgress,
    ProcessedTracker,
    Progress,
    ReadyCompletedTracker,
    StartedTracker,
)


def test_tracker_reset():
    p = StartedTracker(ready=1, started=2)
    p.reset()
    assert p == StartedTracker()


def test_tracker_reset_on_restart():
    t = StartedTracker(ready=3, started=3, completed=2)
    t.reset_on_restart()
    assert t == StartedTracker(ready=2, started=2, completed=2)

    t = ProcessedTracker(ready=4, started=4, processed=3, completed=2)
    t.reset_on_restart()
    assert t == ProcessedTracker(ready=2, started=2, processed=2, completed=2)


@pytest.mark.parametrize("attr", ("ready", "started", "processed", "completed"))
def test_progress_increment(attr):
    p = Progress()
    fn = getattr(p, f"increment_{attr}")
    fn()
    expected = ProcessedTracker(**{attr: 1})
    assert p.total == expected
    assert p.current == expected


def test_progress_from_defaults():
    actual = Progress.from_defaults(StartedTracker, completed=5)
    expected = Progress(total=StartedTracker(completed=5), current=StartedTracker(completed=5))
    assert actual == expected


def test_progress_increment_sequence():
    """Test sequence for incrementing."""
    batch = Progress()

    batch.increment_ready()
    assert batch.total == ProcessedTracker(ready=1)
    assert batch.current == ProcessedTracker(ready=1)

    batch.increment_started()
    assert batch.total == ProcessedTracker(ready=1, started=1)
    assert batch.current == ProcessedTracker(ready=1, started=1)

    batch.increment_processed()
    assert batch.total == ProcessedTracker(ready=1, started=1, processed=1)
    assert batch.current == ProcessedTracker(ready=1, started=1, processed=1)

    batch.increment_completed()
    assert batch.total == ProcessedTracker(ready=1, started=1, processed=1, completed=1)
    assert batch.current == ProcessedTracker(ready=1, started=1, processed=1, completed=1)


def test_progress_raises():
    with pytest.raises(ValueError, match="instances should be of the same class"):
        Progress(ReadyCompletedTracker(), ProcessedTracker())

    p = Progress(ReadyCompletedTracker(), ReadyCompletedTracker())
    with pytest.raises(TypeError, match="ReadyCompletedTracker` doesn't have a `started` attribute"):
        p.increment_started()
    with pytest.raises(TypeError, match="ReadyCompletedTracker` doesn't have a `processed` attribute"):
        p.increment_processed()


def test_optimizer_progress_default_factory():
    """Ensure that the defaults are created appropriately.

    If `default_factory` was not used, the default would be shared between instances.
    """
    p1 = OptimizerProgress()
    p2 = OptimizerProgress()
    p1.step.increment_completed()
    assert p1.step.total.completed == p1.step.current.completed
    assert p1.step.total.completed == 1
    assert p2.step.total.completed == 0


def test_deepcopy():
    _ = deepcopy(BaseProgress())
    _ = deepcopy(Progress())
    _ = deepcopy(ProcessedTracker())
