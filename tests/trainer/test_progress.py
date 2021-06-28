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
import os

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.progress import FitLoopProgress, LoopProgress, Progress, Tracker, TrainingLoopProgress
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class CustomException(BaseException):
    pass


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


@pytest.mark.parametrize("use_multiple_optimizers", [False, True])
@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
def test_progress_tracking(use_multiple_optimizers, accumulate_grad_batches, tmpdir):

    class TestModel(BoringModel):

        def __init__(self):
            super().__init__()
            if use_multiple_optimizers:
                self.configure_optimizers = self.configure_optimizers_3
            self.should_fail = True

        def training_step(self, batch, batch_idx, optimizer_idx: int = None):
            # breaking on global_step 4
            if self.should_fail and self.trainer.current_epoch == 1 and batch_idx == 1 and optimizer_idx == (
                1 if use_multiple_optimizers else None
            ):
                raise CustomException
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

    chk = ModelCheckpoint(dirpath=tmpdir, filename=str(use_multiple_optimizers), save_last=True)
    chk.last_model_path = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=3,
        limit_val_batches=0,
        callbacks=chk,
        accumulate_grad_batches=accumulate_grad_batches,
        resume_from_checkpoint=None,
    )

    # simulate random failure in training_step
    try:
        trainer.fit(model)
    except CustomException:
        pass

    assert isinstance(trainer.fit_loop.progress, FitLoopProgress)
    assert isinstance(trainer.fit_loop.epoch_loop.progress, TrainingLoopProgress)
    assert trainer.fit_loop.epoch_loop.progress is trainer.fit_loop.progress.train

    pr = trainer.fit_loop.epoch_loop.progress

    assert pr.epoch.total == Tracker(ready=2, started=2, processed=1, completed=1)
    assert pr.epoch.current == Tracker(ready=2, started=2, processed=1, completed=1)

    pr = trainer.fit_loop.epoch_loop.progress

    assert pr.batch.total == Tracker(ready=5, started=5, processed=4, completed=4)
    assert pr.batch.current == Tracker(ready=2, started=2, processed=1, completed=1)

    num_optimizers = 3 if use_multiple_optimizers else 1

    optimization = pr.epoch.optimization

    total = (4 * num_optimizers + (1 if use_multiple_optimizers else 0)) // accumulate_grad_batches

    # we raised expection on the first optimizer
    current = (1 if use_multiple_optimizers else 0)

    if accumulate_grad_batches == 2 and use_multiple_optimizers:
        total += 1

    assert optimization.optimizer.total == Tracker(ready=total + 1, started=total, processed=None, completed=total)
    assert optimization.optimizer.current == Tracker(ready=current + 1, started=current, processed=None, completed=current)

    if accumulate_grad_batches == 2:
        # that's weird ! todo (tchaton) investigate this
        total = (9 if use_multiple_optimizers else 3)
        current = 0 # same there.

    assert optimization.zero_grad.total == Tracker(ready=total, started=total, processed=None, completed=total)
    assert optimization.zero_grad.current == Tracker(ready=current, started=current, processed=None, completed=current)

    # for multiple optimizers: 4 batches + 1 on epoch
    total = (5 if use_multiple_optimizers else 1) // accumulate_grad_batches

    if accumulate_grad_batches == 2:
        total += 1

    assert optimization.scheduler.total == Tracker(ready=total, started=None, processed=None, completed=total)
    assert optimization.scheduler.current == Tracker(ready=0, started=None, processed=None, completed=0)

    assert pr.batch.optimizer_idx == (1 if use_multiple_optimizers else 0)

    checkpoint = torch.load(trainer.checkpoint_callback.last_model_path)
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 4 // accumulate_grad_batches

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        limit_train_batches=3,
        limit_val_batches=0,
        resume_from_checkpoint=chk.last_model_path,
        accumulate_grad_batches=accumulate_grad_batches
    )

    model.should_fail = False
    trainer.fit(model)

    pr = trainer.fit_loop.epoch_loop.progress

    assert pr.epoch.total == Tracker(ready=3, started=3, processed=3, completed=3)
    assert pr.epoch.current == Tracker(ready=2, started=2, processed=2, completed=2)

    assert pr.batch.total == Tracker(ready=9, started=9, processed=9, completed=9)
    assert pr.batch.current == Tracker(ready=3, started=3, processed=3, completed=3)

    optimization = pr.epoch.optimization

    if accumulate_grad_batches == 2:
        total = 2 * 3 * (3 if use_multiple_optimizers else 1)
    else:
        total = (3 * 3 * (3 if use_multiple_optimizers else 1))
    current = (3 if use_multiple_optimizers else 1)

    assert optimization.optimizer.total == Tracker(ready=total, started=total, processed=None, completed=total)
    assert optimization.optimizer.current == Tracker(ready=current, started=current, processed=None, completed=current)

    assert optimization.zero_grad.total == Tracker(ready=total, started=total, processed=None, completed=total)
    assert optimization.zero_grad.current == Tracker(ready=current, started=current, processed=None, completed=current)

    # for multiple optimizers: 4 batches + 1 on epoch
    if accumulate_grad_batches == 2:
        total = (2 * 3 + 3 if use_multiple_optimizers else 3)
    else:
        total = (3 * 3 + 3 if use_multiple_optimizers else 3)
    current = (2 if use_multiple_optimizers else 1)
    assert optimization.scheduler.total == Tracker(ready=total, started=None, processed=None, completed=total)
    assert optimization.scheduler.current == Tracker(ready=current, started=None, processed=None, completed=current)


def test_progress_tracking_validation_multiple_datasets(tmpdir):

    class ValidationModel(BoringModel):

        def __init__(self):
            super().__init__()

        def validation_step(self, batch, batch_idx, dataloader_idx):
            if self.trainer.fit_loop.epoch_loop.batch_idx == 3 and batch_idx == 1 and dataloader_idx == 1:
                raise CustomException
            return super().validation_step(batch, batch_idx)

        def val_dataloader(self):
            return [super().val_dataloader(), super().val_dataloader(), super().val_dataloader()]

    model = ValidationModel()
    model.validation_epoch_end = None

    chk = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    chk.last_model_path = None
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=3,
        callbacks=chk,
        resume_from_checkpoint=None,
        val_check_interval=2,
        num_sanity_val_steps=0,
    )

    # simulate random failure in training_step
    try:
        trainer.fit(model)
    except CustomException:
        pass

    pr = trainer.fit_loop.val_loop.progress

    assert isinstance(pr, LoopProgress)
    assert pr.epoch.total == Tracker(ready=2, started=2, processed=1, completed=1)
    assert pr.epoch.current == Tracker(ready=1, started=1, processed=0, completed=0)

    # 3 dataloaders with 3 samples for batch_idx == 1 + first dataloader on batch_idx == 1 + failure on batch_idx = 1
    current = 2
    total = 3 * 3 + 3 + current
    assert pr.batch.total == Tracker(ready=total, started=total, processed=total - 1, completed=total - 1)
    assert pr.batch.current == Tracker(ready=current, started=current, processed=current - 1, completed=current - 1)

    assert pr.dataloader_idx == 1

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=3,
        callbacks=chk,
        resume_from_checkpoint=chk.last_model_path,
        val_check_interval=2,
        num_sanity_val_steps=0,  # TODO (tchaton) This fails when increasing to 1
    )

    trainer.fit(model)

    pr = trainer.fit_loop.epoch_loop.progress

    assert pr.epoch.total == Tracker(ready=1, started=1, processed=1, completed=1)
    assert pr.epoch.current == Tracker(ready=1, started=1, processed=1, completed=1)

    pr = trainer.fit_loop.val_loop.progress

    assert pr.epoch.total == Tracker(ready=2, started=2, processed=2, completed=2)
    assert pr.epoch.current == Tracker(ready=1, started=1, processed=1, completed=1)

    # total = 3 (num validation samples) * 3 (num dataloaders) * 2 (num validation)
    assert pr.batch.total == Tracker(ready=18, started=18, processed=18, completed=18)
    assert pr.batch.current == Tracker(ready=3, started=3, processed=3, completed=3)
    assert pr.dataloader_idx == 2


@RunIf(min_gpus=2, special=True)
def test_ddp_terminate_when_deadlock_is_detected(tmpdir):

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            if batch_idx == 1 and self.trainer.is_global_zero:
                raise CustomException
            return super().training_step(batch, batch_idx)

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=5,
        num_sanity_val_steps=0,
        gpus=2,
        accelerator="ddp",
    )

    # simulate random failure in training_step on rank 0
    try:
        trainer.fit(model)
    except SystemExit:
        pass

    pids = os.getenv("PL_INTERACTIVE_DDP_PROCS", None)
    assert pids is None
