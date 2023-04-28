# Copyright The Lightning AI team.
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

from lightning.pytorch.loops import _FitLoop
from lightning.pytorch.trainer.trainer import Trainer


def test_loops_state_dict():
    trainer = Trainer()

    fit_loop = _FitLoop(trainer)
    state_dict = fit_loop.state_dict()

    new_fit_loop = _FitLoop(trainer)

    new_fit_loop.load_state_dict(state_dict)
    assert fit_loop.state_dict() == new_fit_loop.state_dict()


def test_loops_state_dict_structure():
    trainer = Trainer()
    state_dict = trainer._checkpoint_connector._get_loops_state_dict()
    expected = {
        "fit_loop": {
            "state_dict": {},
            "epoch_loop.state_dict": {"_batches_that_stepped": 0},
            "epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "is_last_batch": False,
            },
            "epoch_loop.scheduler_progress": {
                "total": {"ready": 0, "completed": 0},
                "current": {"ready": 0, "completed": 0},
            },
            "epoch_loop.manual_optimization.state_dict": {},
            "epoch_loop.manual_optimization.optim_step_progress": {
                "total": {"ready": 0, "completed": 0},
                "current": {"ready": 0, "completed": 0},
            },
            "epoch_loop.automatic_optimization.state_dict": {},
            "epoch_loop.automatic_optimization.optim_progress": {
                "optimizer": {
                    "step": {"total": {"ready": 0, "completed": 0}, "current": {"ready": 0, "completed": 0}},
                    "zero_grad": {
                        "total": {"ready": 0, "started": 0, "completed": 0},
                        "current": {"ready": 0, "started": 0, "completed": 0},
                    },
                },
            },
            "epoch_loop.val_loop.state_dict": {},
            "epoch_loop.val_loop.batch_progress": {
                # number of batches across validation runs per epoch across dataloaders
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                # number of batches for this validation run across dataloaders
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "is_last_batch": False,
            },
            "epoch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            },
        },
        "validate_loop": {
            "state_dict": {},
            "batch_progress": {
                # total batches run by `validate` across dataloaders
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                # number of batches run by this `validate` call across dataloaders
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "is_last_batch": False,
            },
        },
        "test_loop": {
            "state_dict": {},
            "batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "is_last_batch": False,
            },
        },
        "predict_loop": {
            "state_dict": {},
            "batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            },
        },
    }
    assert state_dict == expected
