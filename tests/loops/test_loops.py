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
from dataclasses import dataclass
from typing import Any, Dict, Iterator
from unittest import mock
from unittest.mock import ANY

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loops import Loop, TrainingBatchLoop
from pytorch_lightning.trainer.progress import BaseProgress
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf


class NestedLoop(Loop):
    def __init__(self):
        super().__init__()
        self.child_loop0 = None
        self.child_loop1 = None

    @property
    def done(self) -> bool:
        return False

    def connect(self, child0, child1):
        self.child_loop0 = child0
        self.child_loop1 = child1

    def reset(self) -> None:
        pass

    def advance(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("loop_name", ["fit_loop", "validate_loop", "test_loop", "predict_loop"])
def test_connect_loops_direct(loop_name):
    """Test Trainer referenes in loops on assignment."""
    loop = NestedLoop()
    assert loop.trainer is None

    trainer = Trainer()

    # trainer.loop = loop
    setattr(trainer, loop_name, loop)
    assert loop.trainer is trainer


def test_connect_loops_recursive():
    """Test Trainer references in a nested loop assigned to a Trainer."""
    main_loop = NestedLoop()
    child0 = NestedLoop()
    child1 = NestedLoop()
    main_loop.connect(child0, child1)
    assert main_loop.trainer is None
    assert main_loop.child_loop0.trainer is None

    trainer = Trainer()
    trainer.fit_loop = main_loop
    assert child0.trainer is trainer
    assert child1.trainer is trainer


def test_connect_subloops(tmpdir):
    """Test connecting individual subloops by calling `trainer.x.y.connect()`"""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)

    epoch_loop = trainer.fit_loop.epoch_loop
    new_batch_loop = TrainingBatchLoop()
    epoch_loop.connect(batch_loop=new_batch_loop)
    assert epoch_loop.batch_loop is new_batch_loop
    assert new_batch_loop.trainer is None

    trainer.fit(model)
    assert new_batch_loop.trainer is trainer


class CustomException(Exception):
    pass


def test_loop_restore():
    class Simple(Loop):
        def __init__(self, dataset: Iterator):
            super().__init__()
            self.dataset = dataset

        @property
        def skip(self) -> bool:
            return False

        @property
        def done(self) -> bool:
            return self.iteration_count > len(self.dataset)

        def reset(self) -> None:
            self.iter_dataset = iter(self.dataset)
            if self.restarting:
                for _ in range(self.iteration_count):
                    next(self.iter_dataset)
                self.iteration_count += 1
            else:
                self.outputs = []

        def advance(self) -> None:
            value = next(self.iter_dataset)

            if self.iteration_count == 5:
                raise CustomException

            self.outputs.append(value)

        def state_dict(self) -> Dict:
            return {"iteration_count": self.iteration_count, "outputs": self.outputs}

        def load_state_dict(self, state_dict: Dict) -> None:
            self.iteration_count = state_dict["iteration_count"]
            self.outputs = state_dict["outputs"]

    trainer = Trainer()

    data = range(10)
    loop = Simple(data)
    loop.trainer = trainer
    try:
        loop.run()
        state_dict = {}
    except CustomException:
        state_dict = loop.state_dict()

    loop = Simple(data)
    loop.trainer = trainer
    loop.load_state_dict(state_dict)
    loop.restarting = True
    loop.run()

    assert not loop.restarting
    assert loop.outputs == list(range(10))


def test_loop_hierarchy():
    @dataclass
    class SimpleProgress(BaseProgress):
        increment: int = 0

    class Simple(Loop):
        def __init__(self, a):
            super().__init__()
            self.a = a
            self.progress = SimpleProgress()

        def advance(self, *args: Any, **kwargs: Any) -> None:
            loop = getattr(self, "loop_child", None)
            if not loop:
                return
            loop.run()

        def on_advance_end(self):
            self.progress.increment += 1

        @property
        def done(self) -> bool:
            return self.progress.increment > 0

        def reset(self) -> None:
            ...

        def on_save_checkpoint(self) -> Dict:
            return {"a": self.a}

        def on_load_checkpoint(self, state_dict: Dict) -> None:
            self.a = state_dict["a"]

    loop_parent = Simple(1)
    loop_child = Simple(2)
    loop_parent.loop_child = loop_child

    # check the trainer reference is propagated
    loop_parent.trainer = Trainer()
    assert loop_child.trainer is loop_parent.trainer

    state_dict = loop_parent.state_dict()
    assert state_dict == {
        "state_dict": {"a": 1},
        "progress": {"increment": 0},
        "loop_child.state_dict": {"a": 2},
        "loop_child.progress": {"increment": 0},
    }

    state_dict["loop_child.state_dict"]["a"] = 3
    # check restarting after `load_state_dict`
    loop_parent.load_state_dict(state_dict)
    assert loop_parent.restarting

    loop_parent.run()

    # check the new state after `run`
    state_dict = loop_parent.state_dict()
    assert state_dict == {
        "state_dict": {"a": 1},
        "progress": {"increment": 1},
        "loop_child.state_dict": {"a": 3},
        "loop_child.progress": {"increment": 1},
    }

    loop_parent_copy = deepcopy(loop_parent)
    assert loop_parent_copy.state_dict() == loop_parent.state_dict()

    assert loop_parent_copy.on_save_checkpoint() == state_dict["state_dict"]
    assert loop_parent_copy.loop_child.on_save_checkpoint() == state_dict["loop_child.state_dict"]

    loop_parent = Simple(1)
    loop_child = Simple(2)
    loop_parent.loop_child = loop_child
    loop_parent.load_state_dict(state_dict)
    assert loop_parent.progress.increment == 1
    assert loop_parent.loop_child.progress.increment == 1

    del loop_parent.loop_child
    state_dict = loop_parent.state_dict()
    assert state_dict == {"state_dict": {"a": 1}, "progress": {"increment": 1}}


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.parametrize("stop_epoch", (1, 2))
@pytest.mark.parametrize("stop_batch", (1, 2))
@pytest.mark.parametrize("n_dataloaders,stop_dataloader", [(2, 0), (2, 1), (3, 2)])
@RunIf(min_torch="1.7.0")
def test_loop_restart_progress_multiple_dataloaders(tmpdir, n_dataloaders, stop_dataloader, stop_epoch, stop_batch):
    n_batches = 5
    n_epochs = 3

    class ValidationModel(BoringModel):
        def __init__(self):
            super().__init__()

        def validation_step(self, batch, batch_idx, dataloader_idx):
            if self.current_epoch == stop_epoch and batch_idx == stop_batch and dataloader_idx == stop_dataloader:
                raise CustomException
            return super().validation_step(batch, batch_idx)

        def val_dataloader(self):
            return [super(ValidationModel, self).val_dataloader() for _ in range(n_dataloaders)]

    model = ValidationModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=n_epochs,
        limit_train_batches=1,
        limit_val_batches=n_batches,
        num_sanity_val_steps=0,
    )

    # simulate a failure
    try:
        trainer.fit(model)
    except CustomException:
        pass

    ckpt_path = str(tmpdir / ".pl_auto_save.ckpt")
    checkpoint = torch.load(ckpt_path)["loops"]["fit_loop"]

    total_dataloader = stop_epoch * n_dataloaders + stop_dataloader
    expected = {
        "total": {"ready": total_dataloader + 1, "started": None, "processed": None, "completed": total_dataloader},
        "current": {"ready": stop_dataloader + 1, "started": None, "processed": None, "completed": stop_dataloader},
    }
    assert checkpoint["epoch_loop.val_loop.dataloader_progress"] == expected

    trainer.fit_loop.load_state_dict(checkpoint, restart_progress=False)

    # `nbe_`: non-breaking epoch, as in, no exception will be raised. `be_`: breaking epoch
    nbe_total_val_batch = stop_epoch * n_dataloaders * n_batches
    be_total_val_batch = stop_dataloader * n_batches + stop_batch
    total_val_batch = nbe_total_val_batch + be_total_val_batch
    expected = {
        "total": {
            "ready": total_val_batch + 1,
            "started": total_val_batch + 1,
            "processed": total_val_batch,
            "completed": total_val_batch,
        },
        "current": {
            "ready": stop_batch + 1,
            "started": stop_batch + 1,
            "processed": stop_batch,
            "completed": stop_batch,
        },
    }
    assert trainer.fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress.state_dict() == expected

    trainer.fit_loop.load_state_dict(checkpoint)
    expected = {
        "total": {
            "ready": total_val_batch + 1,
            "started": total_val_batch + 1,
            "processed": total_val_batch,
            "completed": total_val_batch,
        },
        "current": {"ready": stop_batch, "started": stop_batch, "processed": stop_batch, "completed": stop_batch},
    }
    assert trainer.fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress.state_dict() == expected


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.parametrize("accumulate_grad_batches", (1, 2, 3))
@pytest.mark.parametrize("n_optimizers", (1, 3, 5))
@pytest.mark.parametrize("stop_epoch", (1, 2))
@pytest.mark.parametrize("stop_batch", (1, 2))
@pytest.mark.parametrize("stop_optimizer", (1, 2))
@RunIf(min_torch="1.7.0")
def test_loop_state_on_exception(accumulate_grad_batches, stop_epoch, stop_batch, stop_optimizer, n_optimizers, tmpdir):
    stop_optimizer = stop_optimizer if stop_optimizer < n_optimizers else 0
    n_epochs = 3
    n_batches = 3

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            if n_optimizers > 1:
                self.configure_optimizers = self.configure_optimizers_multiple

        def training_step(self, batch, batch_idx, optimizer_idx=0):
            if self.trainer.current_epoch == stop_epoch and batch_idx == stop_batch and optimizer_idx == stop_optimizer:
                raise CustomException
            return super().training_step(batch, batch_idx)

        def configure_optimizers_multiple(self):
            optimizers = [torch.optim.Adam(self.layer.parameters(), lr=0.1) for _ in range(n_optimizers)]

            lr_scheduler_0 = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=1)
            lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizers[1], step_size=1)
            # no scheduler for optimizer_2
            lr_schedulers = [lr_scheduler_0, {"scheduler": lr_scheduler_1, "interval": "step"}]

            return optimizers, lr_schedulers

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=n_epochs,
        limit_train_batches=n_batches,
        limit_val_batches=0,
        accumulate_grad_batches=accumulate_grad_batches,
        progress_bar_refresh_rate=0,
        logger=False,
        checkpoint_callback=False,
    )

    # simulate a failure
    try:
        trainer.fit(model)
    except CustomException:
        pass

    ckpt_path = str(tmpdir / ".pl_auto_save.ckpt")
    checkpoint = torch.load(ckpt_path)

    optim_progress = trainer.fit_loop.epoch_loop.batch_loop.optim_progress
    sch_progress = trainer.fit_loop.epoch_loop.scheduler_progress

    # `nbe_`: non-breaking epoch, as in, no exception will be raised. `be_`: breaking epoch
    nbe_batches_completed = stop_epoch * n_batches
    be_batches_completed = stop_batch
    be_batches_ready = stop_batch + 1
    # lightning applies leftover accumulated gradients when the epoch ends
    has_leftover_accumulation_batches = n_batches % accumulate_grad_batches != 0
    # number of batches that will call `optimizer.step()` during non-breaking and breaking epochs
    nbe_stepping_batches = nbe_batches_completed // accumulate_grad_batches
    be_stepping_batches = be_batches_completed // accumulate_grad_batches

    nbe_total_opt_steps = (nbe_stepping_batches + has_leftover_accumulation_batches) * n_optimizers
    does_last_be_batch_step = be_batches_ready % accumulate_grad_batches == 0 or has_leftover_accumulation_batches
    be_total_opt_steps = be_stepping_batches * n_optimizers + does_last_be_batch_step * stop_optimizer
    assert optim_progress.optimizer_steps == nbe_total_opt_steps + be_total_opt_steps
    assert optim_progress.optimizer.step.current.completed == be_total_opt_steps
    has_opt_stepped_in_be = stop_batch + 1 >= accumulate_grad_batches

    nbe_total_zero_grad = (nbe_stepping_batches + has_leftover_accumulation_batches) * n_optimizers
    does_last_be_batch_zero_grad = be_batches_completed % accumulate_grad_batches == 0
    # `max` because the first batch always zero-grads
    be_total_zero_grad = max(1, be_stepping_batches) * n_optimizers + stop_optimizer * does_last_be_batch_zero_grad
    assert optim_progress.optimizer.zero_grad.total.completed == nbe_total_zero_grad + be_total_zero_grad
    assert optim_progress.optimizer.zero_grad.current.completed == be_total_zero_grad

    nbe_sch_steps = stop_epoch
    be_sch_steps = 0  # the current epoch did not complete
    if n_optimizers > 1:
        # assumes that the scheduler config is unchanged
        # `* 1` because there is only one step-level scheduler
        nbe_sch_steps = stop_epoch + nbe_stepping_batches + has_leftover_accumulation_batches * 1
        # `0 +` for the epoch-level scheduler
        be_sch_steps = 0 + be_stepping_batches
    assert sch_progress.total.completed == nbe_sch_steps + be_sch_steps
    assert sch_progress.current.completed == be_sch_steps

    expected = {
        "state_dict": ANY,
        "epoch_progress": {
            "total": {
                "ready": stop_epoch + 1,
                "started": stop_epoch + 1,
                "processed": stop_epoch,
                "completed": stop_epoch,
            },
            "current": {
                "ready": stop_epoch + 1,
                "started": stop_epoch + 1,
                "processed": stop_epoch,
                "completed": stop_epoch,
            },
        },
        "epoch_loop.state_dict": ANY,
        "epoch_loop.batch_progress": {
            "total": {
                "ready": nbe_batches_completed + be_batches_completed + 1,
                "started": nbe_batches_completed + be_batches_completed + 1,
                "processed": nbe_batches_completed + be_batches_completed,
                "completed": nbe_batches_completed + be_batches_completed,
            },
            "current": {
                "ready": stop_batch + 1,
                "started": stop_batch + 1,
                "processed": stop_batch,
                "completed": stop_batch,
            },
        },
        "epoch_loop.scheduler_progress": {
            "total": {
                "ready": nbe_sch_steps + be_sch_steps,
                "started": None,
                "processed": None,
                "completed": nbe_sch_steps + be_sch_steps,
            },
            "current": {"ready": be_sch_steps, "started": None, "processed": None, "completed": be_sch_steps},
        },
        "epoch_loop.batch_loop.state_dict": ANY,
        "epoch_loop.batch_loop.optim_progress": {
            "optimizer_idx": stop_optimizer,
            "optimizer": {
                "step": {
                    "total": {
                        "ready": nbe_total_opt_steps + be_total_opt_steps + has_opt_stepped_in_be,
                        "started": None,
                        "processed": None,
                        "completed": nbe_total_opt_steps + be_total_opt_steps,
                    },
                    "current": {
                        "ready": be_total_opt_steps + has_opt_stepped_in_be,
                        "started": None,
                        "processed": None,
                        "completed": be_total_opt_steps,
                    },
                },
                "zero_grad": {
                    "total": {
                        "ready": nbe_total_zero_grad + be_total_zero_grad,
                        "started": nbe_total_zero_grad + be_total_zero_grad,
                        "processed": None,
                        "completed": nbe_total_zero_grad + be_total_zero_grad,
                    },
                    "current": {
                        "ready": be_total_zero_grad,
                        "started": be_total_zero_grad,
                        "processed": None,
                        "completed": be_total_zero_grad,
                    },
                },
            },
        },
        "epoch_loop.val_loop.state_dict": ANY,
        "epoch_loop.val_loop.dataloader_progress": ANY,
        "epoch_loop.val_loop.epoch_loop.state_dict": ANY,
        "epoch_loop.val_loop.epoch_loop.batch_progress": ANY,
    }
    assert checkpoint["loops"]["fit_loop"] == expected

    trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"], restart_progress=False)
    assert trainer.fit_loop.state_dict() == checkpoint["loops"]["fit_loop"]

    trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"])
    state_dict = trainer.fit_loop.state_dict()
    assert state_dict != checkpoint["loops"]["fit_loop"]
    assert state_dict["epoch_progress"]["total"]["started"] == stop_epoch + 1
    assert state_dict["epoch_progress"]["current"]["started"] == stop_epoch
