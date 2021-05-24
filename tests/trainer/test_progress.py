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

from pytorch_lightning.trainer.progress import LoopProgress, Progress, Tracker


def test_progress_geattr_setattr():
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


def test_loop_progress_increment_epoch():
    p = LoopProgress()
    p.increment_epoch_completed()
    p.increment_epoch_completed()
    assert p.epoch.total == Tracker(completed=2)
    assert p.epoch.current == Tracker()
    assert p.batch.current == Tracker()


def test_loop_progress_increment_sequence():
    """ Test sequences for incrementing batches reads and epochs. """
    p = LoopProgress(batch=Progress(total=Tracker(started=None)))

    p.batch.increment_ready()
    assert p.batch.total == Tracker(ready=1, started=None)
    assert p.batch.current == Tracker(ready=1)

    p.batch.increment_started()
    assert p.batch.total == Tracker(ready=1, started=None)
    assert p.batch.current == Tracker(ready=1)

    p.batch.increment_processed()
    assert p.batch.total == Tracker(ready=1, started=None, processed=1)
    assert p.batch.current == Tracker(ready=1, processed=1)

    p.batch.increment_completed()
    assert p.batch.total == Tracker(ready=1, started=None, processed=1, completed=1)
    assert p.batch.current == Tracker(ready=1, processed=1, completed=1)

    assert p.epoch.total == Tracker()
    assert p.epoch.current == Tracker()
    p.increment_epoch_completed()
    assert p.batch.total == Tracker(ready=1, started=None, processed=1, completed=1)
    assert p.batch.current == Tracker()
    assert p.epoch.total == Tracker(completed=1)
    assert p.epoch.current == Tracker()

    p.batch.increment_ready()
    assert p.batch.total == Tracker(ready=2, started=None, processed=1, completed=1)
    assert p.batch.current == Tracker(ready=1)
    assert p.epoch.total == Tracker(completed=1)
    assert p.epoch.current == Tracker()

    p.reset_on_epoch()
    assert p.batch.total == Tracker(ready=2, started=None, processed=1, completed=1)
    assert p.batch.current == Tracker()
    assert p.epoch.total == Tracker(completed=1)
    assert p.epoch.current == Tracker()
