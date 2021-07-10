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
from copy import deepcopy
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.progress import (
    BatchProgress,
    EpochLoopProgress,
    EpochProgress,
    OptimizerProgress,
    Progress,
    Tracker,
)
from tests.helpers import BoringModel


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


def test_epoch_loop_progress_serialization():
    loop = EpochLoopProgress()
    loop.epoch.dataloader_idx = 1
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
            'dataloader_idx': 1
        }
    }
    # yapf: enable

    new_loop = EpochLoopProgress.from_state_dict(state_dict)
    assert loop == new_loop


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
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

    #######################
    # VALIDATE CHECKPOINT #
    #######################

    checkpoint = torch.load(trainer.checkpoint_callback.last_model_path)

    num_optimizers = 3 if use_multiple_optimizers else 1

    # 4 optimizer steps because breaking on the second batch of the second epoch (3 + 1)
    total_optimizer_step = (4 * num_optimizers + (1 if use_multiple_optimizers else 0)) // accumulate_grad_batches

    # we raised expection on the first optimizer
    current_optimize_step = (1 if use_multiple_optimizers else 0)

    if accumulate_grad_batches == 2 and use_multiple_optimizers:
        total_optimizer_step += 1

    total_optimizer_zero_grad = total_optimizer_step
    current_optimizer_zero_grad = current_optimize_step

    if accumulate_grad_batches == 2:
        # that's weird ! todo (tchaton) investigate this
        total_optimizer_zero_grad = (9 if use_multiple_optimizers else 3)
        current_optimizer_zero_grad = 0  # same there.

    total_scheduler_step = (5 if use_multiple_optimizers else 1) // accumulate_grad_batches

    current_scheduler_step = 0

    if accumulate_grad_batches == 2:
        total_scheduler_step += 1

    optimizer_idx = (1 if use_multiple_optimizers else 0)

    # yapf: disable
    expected = {
        "state_dict": {},
        "epoch_loop.state_dict": {},
        "epoch_loop.progress": {
            "total": {"ready": 2, "started": 2, "processed": 1, "completed": 1},
            "current": {"ready": 2, "started": 2, "processed": 1, "completed": 1},
            "should_check_val": False,
        },
        "epoch_loop.batch_loop.state_dict": {},
        "epoch_loop.batch_loop.progress": {
            "total": {"ready": 5, "started": 5, "processed": 4, "completed": 4},
            "current": {"ready": 2, "started": 2, "processed": 1, "completed": 1},
        },
        "epoch_loop.batch_loop.optim_progress": {
            "optimizer_idx": optimizer_idx,
            "optimizer": {
                "step": {
                    "total": {
                        "ready": total_optimizer_step + 1,
                        "started": total_optimizer_step,
                        "processed": None,
                        "completed": total_optimizer_step
                    },
                    "current": {
                        "ready": current_optimize_step + 1,
                        "started": current_optimize_step,
                        "processed": None,
                        "completed": current_optimize_step,
                    },
                },
                "zero_grad": {
                    "total": {
                        "ready": total_optimizer_zero_grad,
                        "started": total_optimizer_zero_grad,
                        "processed": None,
                        "completed": total_optimizer_zero_grad
                    },
                    "current": {
                        "ready": current_optimizer_zero_grad,
                        "started": current_optimizer_zero_grad,
                        "processed": None,
                        "completed": current_optimizer_zero_grad,
                    },
                },
            },
            "scheduler": {
                "total": {
                    "ready": total_scheduler_step,
                    "started": None,
                    "processed": None,
                    "completed": total_scheduler_step
                },
                "current": {
                    "ready": current_scheduler_step,
                    "started": None,
                    "processed": None,
                    "completed": current_scheduler_step
                },
            },
        },
        "epoch_loop.val_loop.state_dict": {},
        "epoch_loop.val_loop.progress": {
            "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            "dataloader_idx": 0,
        },
        "epoch_loop.val_loop.epoch_loop.state_dict": {},
        "epoch_loop.val_loop.epoch_loop.progress": {
            "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
        },
    }
    # yapf: enable

    assert checkpoint["loops"]["fit_loop"] == expected


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
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

    pr = trainer.fit_loop.epoch_loop.val_loop.progress

    assert pr.total == Tracker(ready=2, started=2, processed=1, completed=1)
    assert pr.current == Tracker(ready=1, started=1, processed=0, completed=0)
    assert pr.dataloader_idx == 1

    assert trainer.fit_loop.epoch_loop.progress.should_check_val

    pr = trainer.fit_loop.epoch_loop.val_loop.epoch_loop.progress

    # 3 dataloaders with 3 samples for batch_idx == 1 + first dataloader on batch_idx == 1 + failure on batch_idx = 1
    current = 2
    total = 3 * 3 + 3 + current
    assert pr.total == Tracker(ready=total, started=total, processed=total - 1, completed=total - 1)
    assert pr.current == Tracker(ready=current, started=current, processed=current - 1, completed=current - 1)
