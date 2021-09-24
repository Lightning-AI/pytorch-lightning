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
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loops import Loop, TrainingBatchLoop
from pytorch_lightning.trainer.progress import BaseProgress
from tests.helpers import BoringModel, RandomDataset
from tests.helpers.runif import RunIf
from tests.utilities.test_auto_restart import _run_validation_loop_fault_tolerance


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

    with pytest.raises(RuntimeError, match="The loop is not attached to a Trainer"):
        _ = loop.trainer

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

    with pytest.raises(RuntimeError, match="The loop is not attached to a Trainer"):
        _ = main_loop.trainer

    with pytest.raises(RuntimeError, match="The loop is not attached to a Trainer"):
        _ = main_loop.child_loop0.trainer

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

    with pytest.raises(RuntimeError, match="The loop is not attached to a Trainer"):
        _ = new_batch_loop.trainer

    trainer.fit(model)
    assert new_batch_loop.trainer is trainer


class CustomException(Exception):
    pass


def test_loop_restore():
    class Simple(Loop):
        def __init__(self, dataset: Iterator):
            super().__init__()
            self.iteration_count = 0
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

        def on_advance_end(self) -> None:
            self.iteration_count += 1

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


@RunIf(min_torch="1.7.0")
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.parametrize("stop_epoch", (1, 2))
@pytest.mark.parametrize("stop_batch", (1, 2))
@pytest.mark.parametrize("n_dataloaders,stop_dataloader", [(2, 0), (2, 1), (3, 2)])
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
    with pytest.raises(CustomException):
        trainer.fit(model)

    ckpt_path = str(tmpdir / ".pl_auto_save.ckpt")
    checkpoint = torch.load(ckpt_path)["loops"]["fit_loop"]

    total_dataloader = stop_epoch * n_dataloaders + stop_dataloader
    expected = {
        "total": {"ready": total_dataloader + 1, "completed": total_dataloader},
        "current": {"ready": stop_dataloader + 1, "completed": stop_dataloader},
    }
    assert checkpoint["epoch_loop.val_loop.dataloader_progress"] == expected

    trainer.fit_loop.load_state_dict(checkpoint)

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


@RunIf(min_torch="1.7.0")
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.parametrize("accumulate_grad_batches", (1, 2, 3))
@pytest.mark.parametrize("n_optimizers", (1, 3, 5))
@pytest.mark.parametrize("stop_epoch", (1, 2))
@pytest.mark.parametrize("stop_batch", (1, 2))
@pytest.mark.parametrize("stop_optimizer", (1, 2))
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
    with pytest.raises(CustomException):
        trainer.fit(model)

    ckpt_path = str(tmpdir / ".pl_auto_save.ckpt")
    assert os.path.exists(ckpt_path)
    checkpoint = torch.load(ckpt_path)

    optim_progress = trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.optim_progress
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
            "is_last_batch": False,
        },
        "epoch_loop.scheduler_progress": {
            "total": {"ready": nbe_sch_steps + be_sch_steps, "completed": nbe_sch_steps + be_sch_steps},
            "current": {"ready": be_sch_steps, "completed": be_sch_steps},
        },
        "epoch_loop.batch_loop.state_dict": ANY,
        "epoch_loop.batch_loop.manual_loop.state_dict": ANY,
        "epoch_loop.batch_loop.optimizer_loop.state_dict": {},
        "epoch_loop.batch_loop.optimizer_loop.optim_progress": {
            "optimizer_position": stop_optimizer,
            "optimizer": {
                "step": {
                    "total": {
                        "ready": nbe_total_opt_steps + be_total_opt_steps + has_opt_stepped_in_be,
                        "completed": nbe_total_opt_steps + be_total_opt_steps,
                    },
                    "current": {"ready": be_total_opt_steps + has_opt_stepped_in_be, "completed": be_total_opt_steps},
                },
                "zero_grad": {
                    "total": {
                        "ready": nbe_total_zero_grad + be_total_zero_grad,
                        "started": nbe_total_zero_grad + be_total_zero_grad,
                        "completed": nbe_total_zero_grad + be_total_zero_grad,
                    },
                    "current": {
                        "ready": be_total_zero_grad,
                        "started": be_total_zero_grad,
                        "completed": be_total_zero_grad,
                    },
                },
            },
        },
        "epoch_loop.val_loop.state_dict": ANY,
        "epoch_loop.val_loop.dataloader_progress": ANY,
        "epoch_loop.val_loop.epoch_loop.state_dict": ANY,
        "epoch_loop.val_loop.epoch_loop.batch_progress": ANY,
        "epoch_loop.val_loop._results": ANY,
        "epoch_loop._results": ANY,
    }
    assert checkpoint["loops"]["fit_loop"] == expected

    trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"])
    state_dict = trainer.fit_loop.state_dict()

    # need to remove these elements for comparison; comparing with `fit_loop.state_dict()` would require the
    # fit loop to have an iterator, which is only available during training
    checkpoint["loops"]["fit_loop"]["state_dict"]["dataloader_state_dict"] = ANY
    assert state_dict == checkpoint["loops"]["fit_loop"]

    trainer.fit_loop.load_state_dict(checkpoint["loops"]["fit_loop"])
    # test resetting manually, we expect all `ready` counters to be reset to `completed`
    trainer.fit_loop.reset()
    trainer.fit_loop.epoch_loop.reset()
    trainer.fit_loop.epoch_loop.batch_loop.reset()
    trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.reset()
    trainer.fit_loop.epoch_loop.val_loop.reset()
    trainer.fit_loop.epoch_loop.val_loop.epoch_loop.reset()

    epoch_progress = trainer.fit_loop.epoch_progress
    assert epoch_progress.current.ready == stop_epoch
    assert epoch_progress.current.completed == stop_epoch

    batch_progress = trainer.fit_loop.epoch_loop.batch_progress
    assert batch_progress.current.ready == be_batches_completed
    assert batch_progress.current.completed == be_batches_completed

    optim_progress = trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.optim_progress
    assert optim_progress.optimizer.step.current.ready == be_total_opt_steps
    assert optim_progress.optimizer.step.current.completed == be_total_opt_steps
    assert optim_progress.optimizer.zero_grad.current.ready == be_total_zero_grad
    assert optim_progress.optimizer.zero_grad.current.completed == be_total_zero_grad

    state_dict = trainer.fit_loop.state_dict()
    assert state_dict != checkpoint["loops"]["fit_loop"]
    assert state_dict["epoch_progress"]["total"]["started"] == stop_epoch + 1
    assert state_dict["epoch_progress"]["current"]["started"] == stop_epoch


@RunIf(min_torch="1.7.0")
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@pytest.mark.parametrize("n_optimizers", (1, 3, 5))
def test_loop_state_on_complete_run(n_optimizers, tmpdir):
    n_epochs = 3
    n_batches = 3
    accumulate_grad_batches = 1

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            if n_optimizers > 1:
                self.configure_optimizers = self.configure_optimizers_multiple

        def training_step(self, batch, batch_idx, optimizer_idx=0):
            return super().training_step(batch, batch_idx)

        def configure_optimizers_multiple(self):
            optimizers = [torch.optim.Adam(self.layer.parameters(), lr=0.1) for _ in range(n_optimizers)]

            lr_scheduler_0 = torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=1)
            lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizers[1], step_size=1)
            # no scheduler for optimizer_2
            lr_schedulers = [lr_scheduler_0, {"scheduler": lr_scheduler_1, "interval": "step"}]

            return optimizers, lr_schedulers

        def train_dataloader(self):
            # override to test the `is_last_batch` value
            return DataLoader(RandomDataset(32, n_batches))

    model = TestModel()
    model.training_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=n_epochs,
        limit_val_batches=0,
        accumulate_grad_batches=accumulate_grad_batches,
        progress_bar_refresh_rate=0,
        logger=False,
        checkpoint_callback=True,
    )
    trainer.fit(model)

    assert trainer.num_training_batches == n_batches

    ckpt_path = trainer.checkpoint_callback.best_model_path
    assert os.path.exists(ckpt_path)
    checkpoint = torch.load(ckpt_path)

    n_sch_steps_total = n_epochs
    n_sch_steps_current = 1
    if n_optimizers > 1:
        n_sch_steps_total = n_epochs + n_epochs * n_batches
        n_sch_steps_current = n_batches + 1

    expected = {
        "state_dict": ANY,
        "epoch_progress": {
            "total": {
                "ready": n_epochs,
                "started": n_epochs,
                "processed": n_epochs,
                # TODO: the following "-1" offset will be fixed by
                #   https://github.com/PyTorchLightning/pytorch-lightning/pull/8578
                "completed": n_epochs - 1,
            },
            "current": {
                "ready": n_epochs,
                "started": n_epochs,
                "processed": n_epochs,
                # TODO: the following "-1" offset will be fixed by
                #   https://github.com/PyTorchLightning/pytorch-lightning/pull/8578
                "completed": n_epochs - 1,
            },
        },
        "epoch_loop.state_dict": ANY,
        "epoch_loop.batch_progress": {
            "total": {
                "ready": n_epochs * n_batches,
                "started": n_epochs * n_batches,
                "processed": n_epochs * n_batches,
                "completed": n_epochs * n_batches,
            },
            "current": {
                "ready": n_batches,
                "started": n_batches,
                "processed": n_batches,
                "completed": n_batches,
            },
            "is_last_batch": True,
        },
        "epoch_loop.scheduler_progress": {
            "total": {"ready": n_sch_steps_total, "completed": n_sch_steps_total},
            "current": {"ready": n_sch_steps_current, "completed": n_sch_steps_current},
        },
        "epoch_loop.batch_loop.state_dict": ANY,
        "epoch_loop.batch_loop.manual_loop.state_dict": ANY,
        "epoch_loop.batch_loop.optimizer_loop.state_dict": {},
        "epoch_loop.batch_loop.optimizer_loop.optim_progress": {
            "optimizer_position": n_optimizers,
            "optimizer": {
                "step": {
                    "total": {
                        "ready": n_epochs * n_batches * n_optimizers,
                        "completed": n_epochs * n_batches * n_optimizers,
                    },
                    "current": {
                        "ready": n_batches * n_optimizers,
                        "completed": n_batches * n_optimizers,
                    },
                },
                "zero_grad": {
                    "total": {
                        "ready": n_epochs * n_batches * n_optimizers,
                        "started": n_epochs * n_batches * n_optimizers,
                        "completed": n_epochs * n_batches * n_optimizers,
                    },
                    "current": {
                        "ready": n_batches * n_optimizers,
                        "started": n_batches * n_optimizers,
                        "completed": n_batches * n_optimizers,
                    },
                },
            },
        },
        "epoch_loop.val_loop.state_dict": ANY,
        "epoch_loop.val_loop.dataloader_progress": ANY,
        "epoch_loop.val_loop.epoch_loop.state_dict": ANY,
        "epoch_loop.val_loop.epoch_loop.batch_progress": ANY,
        "epoch_loop.val_loop._results": ANY,
        "epoch_loop._results": ANY,
    }
    assert checkpoint["loops"]["fit_loop"] == expected


@RunIf(min_torch="1.7.0")
@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def test_fit_loop_reset(tmpdir):
    """Test that the reset logic in fit- and epoch loop is aware of whether the loop is restarting from a completed
    loop or from a mid-epoch checkpoint."""

    # generate checkpoints at end of epoch and mid-epoch
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmpdir,
        every_n_train_steps=2,
        save_top_k=-1,
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=4,
        num_sanity_val_steps=0,
        max_epochs=2,
        callbacks=[checkpoint_callback],
        logger=False,
        weights_summary=None,
    )
    trainer.fit(model)

    # reset state loaded from a checkpoint from mid-epoch
    mid_epoch_ckpt = torch.load(str(tmpdir / "epoch=0-step=1.ckpt"))
    fit_loop = trainer.fit_loop
    epoch_loop = fit_loop.epoch_loop
    optimizer_loop = epoch_loop.batch_loop.optimizer_loop
    assert not fit_loop.restarting
    assert not epoch_loop.restarting
    assert not optimizer_loop.restarting

    # we load exactly what was saved - no reset yet
    fit_loop.load_state_dict(mid_epoch_ckpt["loops"]["fit_loop"])
    # resetting from a mid-of-epoch checkpoint SHOULD NOT reset the current counters to 0
    fit_loop.reset()
    epoch_loop.reset()
    optimizer_loop.reset()

    assert fit_loop.restarting
    assert fit_loop.epoch_progress.total.ready == 1
    assert fit_loop.epoch_progress.total.completed == 0  # the checkpoint was saved mid epoch
    assert fit_loop.epoch_progress.current.ready == 0
    assert fit_loop.epoch_progress.current.completed == 0

    assert epoch_loop.restarting
    assert epoch_loop.batch_progress.total.ready == 2
    assert epoch_loop.batch_progress.total.processed == 2
    assert epoch_loop.batch_progress.total.completed == 1  # the checkpoint was saved on train_batch_end
    assert epoch_loop.batch_progress.current.ready == 1  # currents get set to the completed value
    assert epoch_loop.batch_progress.current.processed == 1
    assert epoch_loop.batch_progress.current.completed == 1

    assert optimizer_loop.restarting
    assert optimizer_loop.optim_progress.optimizer_position == 1

    # reset state loaded from a checkpoint from the end of an epoch
    end_of_epoch_ckpt = torch.load(str(tmpdir / "epoch=0-step=3.ckpt"))
    fit_loop = trainer.fit_loop
    epoch_loop = fit_loop.epoch_loop
    fit_loop.restarting = False
    epoch_loop.restarting = False
    epoch_loop.val_loop.restarting = False
    epoch_loop.val_loop.epoch_loop.restarting = False
    optimizer_loop.restarting = False

    # we load exactly what was saved - no reset yet
    fit_loop.load_state_dict(end_of_epoch_ckpt["loops"]["fit_loop"])
    # resetting from a end-of-epoch checkpoint SHOULD reset the current counters to 0
    fit_loop.reset()
    epoch_loop.reset()
    epoch_loop.batch_loop.reset()
    epoch_loop.val_loop.reset()
    optimizer_loop.reset()

    assert fit_loop.restarting
    assert fit_loop.epoch_progress.total.ready == 1
    assert fit_loop.epoch_progress.total.completed == 0  # the checkpoint saves before the epoch completes
    assert fit_loop.epoch_progress.current.ready == 0
    assert fit_loop.epoch_progress.current.completed == 0

    assert epoch_loop.restarting
    assert epoch_loop.batch_progress.total.ready == 4
    assert epoch_loop.batch_progress.total.processed == 4
    assert epoch_loop.batch_progress.total.completed == 3  # the checkpoint was saved on train_batch_end
    assert epoch_loop.batch_progress.current.ready == 3  # currents get set to the completed value
    assert epoch_loop.batch_progress.current.processed == 3
    assert epoch_loop.batch_progress.current.completed == 3

    assert optimizer_loop.optim_progress.optimizer_position == 1


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
@RunIf(min_torch="1.7.0")
@pytest.mark.parametrize(
    "dataset_classes",
    [
        [[RandomDataset], [RandomDataset]],
        [[RandomDataset], [RandomDataset, RandomDataset]],
    ],
)
@pytest.mark.parametrize("val_check_interval", [0.5, 1.0])
def test_auto_restart_within_validation_loop(dataset_classes, val_check_interval, tmpdir):
    num_samples = 4
    num_validation_loaders = len(dataset_classes[1])
    trainer, training_step_batches, validation_step_batches = _run_validation_loop_fault_tolerance(
        dataset_classes, tmpdir, val_check_interval
    )

    assert len(training_step_batches) == num_samples
    assert len(validation_step_batches) == num_validation_loaders
    for batch in validation_step_batches.values():
        assert len(batch) == (1 / val_check_interval) * num_samples

    checkpoint_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")
    assert not os.path.exists(checkpoint_path)

    state_dict = trainer.fit_loop.state_dict()

    expected = 2 if val_check_interval == 1.0 else 0
    state_dict["epoch_loop.batch_progress"]["total"] = {
        "ready": 2 + expected,
        "completed": 2 + expected,
        "started": 2 + expected,
        "processed": 2 + expected,
    }

    state_dict["epoch_loop.batch_progress"]["current"] = {
        "ready": 2 + expected,
        "completed": 2 + expected,
        "started": 2 + expected,
        "processed": 2 + expected,
    }

    expected = 2 if val_check_interval == 1.0 else 0
    state_dict["epoch_loop.val_loop.dataloader_progress"]["total"] == {
        "ready": 1 + expected,
        "completed": expected,
    }

    total = (1 / val_check_interval) * num_validation_loaders * num_samples
    state_dict["epoch_loop.val_loop.epoch_loop.batch_progress"]["total"] == {
        "ready": total,
        "completed": total,
    }

    state_dict["epoch_loop.val_loop.epoch_loop.batch_progress"]["current"] == {
        "ready": num_samples,
        "completed": num_samples,
    }

    _, training_step_batches, validation_step_batches = _run_validation_loop_fault_tolerance(
        dataset_classes, tmpdir, val_check_interval, should_fail=True
    )

    checkpoint_path = os.path.join(tmpdir, ".pl_auto_save.ckpt")
    assert os.path.exists(checkpoint_path)

    checkpoint = torch.load(checkpoint_path)["loops"]["fit_loop"]

    shift = 2 if val_check_interval == 1.0 else 0
    assert checkpoint["epoch_loop.batch_progress"]["total"] == {
        "ready": 2 + shift,
        "completed": 2 + shift,
        "started": 2 + shift,
        "processed": 2 + shift,
    }
    assert checkpoint["epoch_loop.batch_progress"]["current"] == {
        "ready": 2 + shift,
        "completed": 2 + shift,
        "started": 2 + shift,
        "processed": 2 + shift,
    }

    total = 5 if num_validation_loaders == 2 else 1
    state_dict["epoch_loop.val_loop.epoch_loop.batch_progress"]["total"] == {
        "ready": total,
        "completed": total,
    }

    state_dict["epoch_loop.val_loop.epoch_loop.batch_progress"]["current"] == {
        "ready": 1,
        "completed": 1,
    }

    trainer, _, _ = _run_validation_loop_fault_tolerance(dataset_classes, tmpdir, val_check_interval, resume=True)

    state_dict = trainer.fit_loop.state_dict()

    total = (1 / val_check_interval) * num_validation_loaders * num_samples
    state_dict["epoch_loop.val_loop.epoch_loop.batch_progress"]["total"] == {
        "ready": total,
        "completed": total,
    }

    state_dict["epoch_loop.val_loop.epoch_loop.batch_progress"]["current"] == {
        "ready": num_samples,
        "completed": num_samples,
    }
