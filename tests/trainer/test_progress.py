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
import torch

from pl_examples.bug_report_model import BoringModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.progress import (
    BatchProgress,
    EpochLoopProgress,
    EpochProgress,
    FitLoopProgress,
    OptimizerProgress,
    Progress,
    Tracker,
    TrainingLoopProgress,
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
    state_dict = fit_loop.state_dict()
    # yapf: disable
    assert state_dict == {
        'epoch': {
            # number of epochs across `fit` calls
            'total': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
            # number of epochs this `fit` call
            'current': {'completed': 0, 'processed': 0, 'ready': 0, 'started': 0},
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


@pytest.mark.parametrize("use_multiple_optimizers", [False, True])
def test_progress_tracking_fit_loop(use_multiple_optimizers, tmpdir):

    class TestModel(BoringModel):

        def __init__(self):
            super().__init__()
            if use_multiple_optimizers:
                self.configure_optimizers = self.configure_optimizers_3

        def training_step(self, batch, batch_idx, optimizer_idx: int = None):
            return super().training_step(batch, batch_idx)

        def configure_optimizers_3(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            optimizer_1 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=1)
            optimizer_2 = torch.optim.Adam(self.layer.parameters(), lr=0.1)
            return [optimizer, optimizer_1, optimizer_2], \
                   [lr_scheduler, {"scheduler": lr_scheduler_1, "interval": "step"}]

    model = TestModel()
    model.training_epoch_end = None

    chk = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_train_batches=3, limit_val_batches=0, callbacks=chk)
    trainer.fit(model)

    assert isinstance(trainer.fit_loop.progress, FitLoopProgress)
    assert isinstance(trainer.fit_loop.epoch_loop.progress, TrainingLoopProgress)
    assert trainer.fit_loop.epoch_loop.progress is trainer.fit_loop.progress.train

    pr = trainer.fit_loop.epoch_loop.progress

    assert pr.epoch.total == Tracker(ready=2, started=2, processed=2, completed=2)
    assert pr.epoch.current == Tracker(ready=2, started=2, processed=2, completed=2)

    pr = trainer.fit_loop.epoch_loop.progress

    assert pr.batch.total == Tracker(ready=6, started=6, processed=6, completed=6)
    assert pr.batch.current == Tracker(ready=3, started=3, processed=3, completed=3)

    num_optimizers = 3 if use_multiple_optimizers else 1
    for _ in range(num_optimizers):

        total = 6 * num_optimizers
        current = 3 * num_optimizers

        pr.epoch.optimization.optimizer.total = Tracker(ready=total, started=total, processed=None, completed=total)
        pr.epoch.optimization.optimizer.current = Tracker(
            ready=current, started=current, processed=None, completed=current
        )

        pr.epoch.optimization.scheduler.total = Tracker(ready=total, started=total, processed=None, completed=total)
        pr.epoch.optimization.scheduler.current = Tracker(
            ready=current, started=current, processed=None, completed=current
        )

        pr.epoch.optimization.zero_grad.total = Tracker(ready=total, started=total, processed=None, completed=total)
        pr.epoch.optimization.zero_grad.current = Tracker(
            ready=current, started=current, processed=None, completed=current
        )

    assert pr.batch.optimizer_idx == (2 if use_multiple_optimizers else 0)

    progress = trainer.fit_loop.progress

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        limit_train_batches=3,
        limit_val_batches=0,
        resume_from_checkpoint=chk.last_model_path
    )

    # TODO(@tchaton): Update this when restore progress is supported.
    trainer.fit_loop.progress = progress
    trainer.fit_loop.epoch_loop.progress = progress.train

    trainer.fit(model)

    pr = trainer.fit_loop.epoch_loop.progress

    assert pr.epoch.total == Tracker(ready=3, started=3, processed=3, completed=3)
    assert pr.epoch.current == Tracker(ready=1, started=1, processed=1, completed=1)
    assert pr.epoch.batch.current == Tracker()
