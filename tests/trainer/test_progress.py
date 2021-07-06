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
    BatchProgress,
    EpochLoopProgress,
    EpochProgress,
    FitLoopProgress,
    OptimizerProgress,
    Progress,
    Tracker,
)


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


def test_epoch_loop_progress_increment_epoch():
    p = EpochLoopProgress()
    p.increment_epoch_completed()
    p.increment_epoch_completed()
    assert p.epoch.total == Tracker(completed=2)
    assert p.epoch.current == Tracker()
    assert p.epoch.batch.current == Tracker()


def test_epoch_loop_progress_increment_sequence():
    """Test sequences for incrementing batches reads and epochs."""
    batch = BatchProgress(total=Tracker(started=None))
    epoch = EpochProgress(batch=batch)
    loop = EpochLoopProgress(epoch=epoch)

    batch.increment_ready()
    assert batch.total == Tracker(ready=1, started=None)
    assert batch.current == Tracker(ready=1)

    batch.increment_started()
    assert batch.total == Tracker(ready=1, started=None)
    assert batch.current == Tracker(ready=1)

    batch.increment_processed()
    assert batch.total == Tracker(ready=1, started=None, processed=1)
    assert batch.current == Tracker(ready=1, processed=1)

    batch.increment_completed()
    assert batch.total == Tracker(ready=1, started=None, processed=1, completed=1)
    assert batch.current == Tracker(ready=1, processed=1, completed=1)

    assert epoch.total == Tracker()
    assert epoch.current == Tracker()
    loop.increment_epoch_completed()
    assert batch.total == Tracker(ready=1, started=None, processed=1, completed=1)
    assert batch.current == Tracker()
    assert epoch.total == Tracker(completed=1)
    assert epoch.current == Tracker()

    batch.increment_ready()
    assert batch.total == Tracker(ready=2, started=None, processed=1, completed=1)
    assert batch.current == Tracker(ready=1)
    assert epoch.total == Tracker(completed=1)
    assert epoch.current == Tracker()

    loop.reset_on_epoch()
    assert batch.total == Tracker(ready=2, started=None, processed=1, completed=1)
    assert batch.current == Tracker()
    assert epoch.total == Tracker(completed=1)
    assert epoch.current == Tracker()


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


def test_fit_loop_progress_serialization():
    fit_loop = FitLoopProgress()
    _ = deepcopy(fit_loop)
    fit_loop.epoch.increment_completed()  # check `TrainingEpochProgress.load_state_dict` calls `super`

    state_dict = fit_loop.state_dict()
    # yapf: disable
    assert state_dict == {
        'epoch': {
            # number of epochs across `fit` calls
            'total': {'completed': 1, 'processed': 0, 'ready': 0, 'started': 0},
            # number of epochs this `fit` call
            'current': {'completed': 1, 'processed': 0, 'ready': 0, 'started': 0},
            'batch': {
                # number of batches across `fit` calls
                'total': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
                # number of batches this epoch
                'current': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
            },
            # `fit` optimization progress
            'optim': {
                # optimizers progress
                'optimizer': {
                    'step': {
                        # `optimizer.step` calls across `fit` calls
                        'total': {'completed': 0, 'processed': None, 'ready': 0, 'started': 0},
                        # `optimizer.step` calls this epoch
                        'current': {'completed': 0, 'processed': None, 'ready': 0, 'started': 0},
                    },
                    'zero_grad': {
                        # `optimizer.zero_grad` calls across `fit` calls
                        'total': {'completed': 0, 'processed': None, 'ready': 0, 'started': 0},
                        # `optimizer.zero_grad` calls this epoch
                        'current': {'completed': 0, 'processed': None, 'ready': 0, 'started': 0},
                    },
                },
                'scheduler': {
                    # `scheduler.step` calls across `fit` calls
                    'total': {'completed': 0, 'processed': None, 'ready': 0, 'started': None},
                    # `scheduler.step` calls this epoch
                    'current': {'completed': 0, 'processed': None, 'ready': 0, 'started': None},
                },
            },
            # `fit` validation progress
            'val': {
                'epoch': {
                    # number of `validation` calls across `fit` calls
                    'total': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
                    # number of `validation` calls this `fit` call
                    'current': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
                    'batch': {
                        # number of batches across `fit` `validation` calls
                        'total': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
                        # number of batches this `fit` `validation` call
                        'current': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
                    },
                }
            },
        }
    }
    # yapf: enable

    new_loop = FitLoopProgress.from_state_dict(state_dict)
    assert fit_loop == new_loop


def test_epoch_loop_progress_serialization():
    loop = EpochLoopProgress()
    _ = deepcopy(loop)
    state_dict = loop.state_dict()

    # yapf: disable
    assert state_dict == {
        'epoch': {
            # number of times `validate` has been called
            'total': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
            # either 0 or 1 as `max_epochs` does not apply to the `validate` loop
            'current': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
            'batch': {
                # number of batches across `validate` calls
                'total': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
                # number of batches this `validate` call
                'current': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
            },
        }
    }
    # yapf: enable

    new_loop = EpochLoopProgress.from_state_dict(state_dict)
    assert loop == new_loop
