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

import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.trainer.progress import BaseProgress
from pytorch_lightning.trainer.trainer import Trainer
from tests.helpers import BoringModel


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
        'state_dict': {
            'a': 1
        },
        'progress': {
            'increment': 0
        },
        'loop_child.state_dict': {
            'a': 2
        },
        'loop_child.progress': {
            'increment': 0
        }
    }

    state_dict["loop_child.state_dict"]["a"] = 3
    # check restarting after `load_state_dict`
    loop_parent.load_state_dict(state_dict)
    assert loop_parent.restarting

    loop_parent.run()

    # check the new state after `run`
    state_dict = loop_parent.state_dict()
    assert state_dict == {
        'state_dict': {
            'a': 1
        },
        'progress': {
            'increment': 1
        },
        'loop_child.state_dict': {
            'a': 3
        },
        'loop_child.progress': {
            'increment': 1
        }
    }

    loop_parent_copy = deepcopy(loop_parent)
    assert loop_parent_copy.state_dict() == loop_parent.state_dict()

    assert loop_parent_copy.on_save_checkpoint() == state_dict['state_dict']
    assert loop_parent_copy.loop_child.on_save_checkpoint() == state_dict['loop_child.state_dict']

    loop_parent = Simple(1)
    loop_child = Simple(2)
    loop_parent.loop_child = loop_child
    loop_parent.load_state_dict(state_dict)
    assert loop_parent.progress.increment == 1
    assert loop_parent.loop_child.progress.increment == 1

    del loop_parent.loop_child
    state_dict = loop_parent.state_dict()
    assert state_dict == {'state_dict': {'a': 1}, 'progress': {'increment': 1}}


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def test_loop_restart_progress_multiple_datasets(tmpdir):
    stop_epoch = stop_batch = stop_dataloader = 1
    n_dataloaders = 3
    n_batches = 3
    n_epochs = 2

    class ValidationModel(BoringModel):

        def __init__(self):
            super().__init__()

        def validation_step(self, batch, batch_idx, dataloader_idx):
            if self.current_epoch == stop_epoch and batch_idx == stop_batch and dataloader_idx == stop_dataloader:
                raise CustomException
            return super().validation_step(batch, batch_idx)

        def val_dataloader(self):
            return [super().val_dataloader()] * n_dataloaders

    model = ValidationModel()
    model.validation_epoch_end = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=n_epochs,
        limit_train_batches=1,
        limit_val_batches=n_batches,
        callbacks=ModelCheckpoint(dirpath=tmpdir, save_last=True),
        num_sanity_val_steps=0,
    )

    # simulate random failure in training_step
    try:
        trainer.fit(model)
    except CustomException:
        pass

    ckpt_path = str(tmpdir / '.pl_auto_save.ckpt')
    checkpoint = torch.load(ckpt_path)["loops"]["fit_loop"]

    total = (n_epochs - 1) * n_dataloaders + stop_dataloader
    expected = {
        "total": {
            "ready": total + 1,
            "started": None,
            "processed": None,
            "completed": total
        },
        "current": {
            "ready": stop_dataloader + 1,
            "started": None,
            "processed": None,
            "completed": stop_dataloader,
        },
    }
    assert checkpoint["epoch_loop.val_loop.dataloader_progress"] == expected

    trainer.fit_loop.load_state_dict(checkpoint, restart_progress=False)
    total = n_dataloaders * n_batches + n_batches + stop_epoch
    expected = {
        "total": {
            "ready": total + 1,
            "started": total + 1,
            "processed": total,
            "completed": total
        },
        "current": {
            "ready": stop_batch + 1,
            "started": stop_batch + 1,
            "processed": stop_batch,
            "completed": stop_batch
        },
    }
    assert trainer.fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress.state_dict() == expected

    trainer.fit_loop.load_state_dict(checkpoint)
    expected = {
        "total": {
            "ready": total,
            "started": total,
            "processed": total,
            "completed": total
        },
        "current": {
            "ready": stop_batch,
            "started": stop_batch,
            "processed": stop_batch,
            "completed": stop_batch
        },
    }
    assert trainer.fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress.state_dict() == expected
