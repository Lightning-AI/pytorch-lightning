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
from unittest.mock import Mock

import pytest

from pytorch_lightning.loops import FitLoop
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_loops_state_dict():
    trainer = Trainer()

    fit_loop = FitLoop()
    with pytest.raises(MisconfigurationException, match="Loop FitLoop should be connected to a"):
        fit_loop.trainer = object()

    fit_loop.trainer = trainer
    state_dict = fit_loop.state_dict()

    new_fit_loop = FitLoop()
    new_fit_loop.trainer = trainer

    new_fit_loop.load_state_dict(state_dict)
    assert fit_loop.state_dict() == new_fit_loop.state_dict()


def test_loops_state_dict_structure():
    trainer = Trainer()
    trainer.train_dataloader = Mock()
    state_dict = trainer.checkpoint_connector._get_loops_state_dict()
    expected = {
        "fit_loop": {
            "state_dict": {},
            "epoch_loop.state_dict": {},
            "epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "is_last_batch": False,
            },
            "epoch_loop.scheduler_progress": {
                "total": {"ready": 0, "completed": 0},
                "current": {"ready": 0, "completed": 0},
            },
            "epoch_loop.batch_loop.state_dict": {},
            "epoch_loop.batch_loop.manual_loop.state_dict": {},
            "epoch_loop.batch_loop.optimizer_loop.state_dict": {},
            "epoch_loop.batch_loop.optimizer_loop.optim_progress": {
                "optimizer": {
                    "step": {"total": {"ready": 0, "completed": 0}, "current": {"ready": 0, "completed": 0}},
                    "zero_grad": {
                        "total": {"ready": 0, "started": 0, "completed": 0},
                        "current": {"ready": 0, "started": 0, "completed": 0},
                    },
                },
                "optimizer_position": 0,
            },
            "epoch_loop.val_loop.state_dict": {},
            "epoch_loop.val_loop.dataloader_progress": {
                "total": {"ready": 0, "completed": 0},
                "current": {"ready": 0, "completed": 0},
            },
            "epoch_loop.val_loop.epoch_loop.state_dict": {},
            "epoch_loop.val_loop.epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "is_last_batch": False,
            },
            "epoch_loop.val_loop._results": {
                "batch": None,
                "batch_size": None,
                "training": False,
                "device": None,
                "items": {},
            },
            "epoch_loop._results": {
                "batch": None,
                "batch_size": None,
                "training": True,
                "device": None,
                "items": {},
            },
            "epoch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            },
        },
        "validate_loop": {
            "state_dict": {},
            "dataloader_progress": {"total": {"ready": 0, "completed": 0}, "current": {"ready": 0, "completed": 0}},
            "epoch_loop.state_dict": {},
            "epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "is_last_batch": False,
            },
            "_results": {
                "batch": None,
                "batch_size": None,
                "training": False,
                "device": None,
                "items": {},
            },
        },
        "test_loop": {
            "state_dict": {},
            "dataloader_progress": {"total": {"ready": 0, "completed": 0}, "current": {"ready": 0, "completed": 0}},
            "epoch_loop.state_dict": {},
            "epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "is_last_batch": False,
            },
            "_results": {
                "batch": None,
                "batch_size": None,
                "training": False,
                "device": None,
                "items": {},
            },
        },
        "predict_loop": {
            "state_dict": {},
            "dataloader_progress": {"total": {"ready": 0, "completed": 0}, "current": {"ready": 0, "completed": 0}},
            "epoch_loop.state_dict": {},
            "epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            },
        },
    }
    assert state_dict == expected
