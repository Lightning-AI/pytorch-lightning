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

from pytorch_lightning.trainer.progress import BaseProgress, OptimizerProgress, Progress, Tracker


def test_progress_getattr_setattr():
    p = Tracker(ready=10, completed=None)
    # can read
    assert p.completed is None
    # can't read non-existing attr
    with pytest.raises(AttributeError, match="object has no attribute 'non_existing_attr'"):
        p.non_existing_attr  # noqa
    # can set new attr
    p.non_existing_attr = 10
    # can't write unused attr
    with pytest.raises(AttributeError, match="'completed' attribute is meant to be unused"):
        p.completed = 10
    with pytest.raises(TypeError, match="unsupported operand type"):
        # default python error, would need to override `__getattribute__`
        # but we want to allow reading the `None` value
        p.completed += 10


def test_progress_reset():
    p = Tracker(ready=1, started=2, completed=None)
    p.reset()
    assert p == Tracker(completed=None)


def test_progress_repr():
    assert repr(Tracker(ready=None, started=None)) == "Tracker(processed=0, completed=0)"


@pytest.mark.parametrize("attr", ("ready", "started", "processed", "completed"))
def test_base_progress_increment(attr):
    p = Progress()
    fn = getattr(p, f"increment_{attr}")
    fn()
    expected = Tracker(**{attr: 1})
    assert p.total == expected
    assert p.current == expected


def test_base_progress_from_defaults():
    actual = Progress.from_defaults(completed=5, started=None)
    expected = Progress(total=Tracker(started=None, completed=5), current=Tracker(started=None, completed=5))
    assert actual == expected


def test_epoch_loop_progress_increment_sequence():
    """Test sequences for incrementing batches reads and epochs."""
    batch = Progress()

    batch.increment_ready()
    assert batch.total == Tracker(ready=1)
    assert batch.current == Tracker(ready=1)

    batch.increment_started()
    assert batch.total == Tracker(ready=1, started=1)
    assert batch.current == Tracker(ready=1, started=1)

    batch.increment_processed()
    assert batch.total == Tracker(ready=1, started=1, processed=1)
    assert batch.current == Tracker(ready=1, started=1, processed=1)

    batch.increment_completed()
    assert batch.total == Tracker(ready=1, started=1, processed=1, completed=1)
    assert batch.current == Tracker(ready=1, started=1, processed=1, completed=1)


def test_optimizer_progress_default_factory():
    """
    Ensure that the defaults are created appropiately. If `default_factory` was not used, the default would
    be shared between instances.
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
    _ = deepcopy(Tracker())
