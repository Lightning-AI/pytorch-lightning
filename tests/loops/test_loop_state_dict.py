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
import torch

from pytorch_lightning.loops import FitLoop
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel


def test_loops_state_dict():
    fit_loop = FitLoop()
    with pytest.raises(MisconfigurationException, match="Loop FitLoop should be connected to a"):
        fit_loop.trainer = object()

    fit_loop.connect(Mock())
    state_dict = fit_loop.state_dict()
    new_fit_loop = FitLoop()
    new_fit_loop.connect(Trainer())
    new_fit_loop.load_state_dict(state_dict)
    assert fit_loop.state_dict() == new_fit_loop.state_dict()


def test_loops_state_dict_structure():
    trainer = Trainer()
    state_dict = trainer.checkpoint_connector._get_loops_state_dict()
    expected = {
        "fit_loop": {
            "state_dict": {},
            "epoch_loop.state_dict": {},
            "epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            },
            "epoch_loop.scheduler_progress": {
                "total": {"ready": 0, "started": None, "processed": None, "completed": 0},
                "current": {"ready": 0, "started": None, "processed": None, "completed": 0},
            },
            "epoch_loop.batch_loop.state_dict": {},
            "epoch_loop.batch_loop.optim_progress": {
                "optimizer": {
                    "step": {
                        "total": {"ready": 0, "started": None, "processed": None, "completed": 0},
                        "current": {"ready": 0, "started": None, "processed": None, "completed": 0},
                    },
                    "zero_grad": {
                        "total": {"ready": 0, "started": 0, "processed": None, "completed": 0},
                        "current": {"ready": 0, "started": 0, "processed": None, "completed": 0},
                    },
                },
                "optimizer_idx": 0,
            },
            "epoch_loop.val_loop.state_dict": {},
            "epoch_loop.val_loop.dataloader_progress": {
                "total": {"ready": 0, "started": None, "processed": None, "completed": 0},
                "current": {"ready": 0, "started": None, "processed": None, "completed": 0},
            },
            "epoch_loop.val_loop.epoch_loop.state_dict": {},
            "epoch_loop.val_loop.epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            },
            "epoch_loop.val_loop._results": {
                "training": False,
                "_minimize": None,
                "_batch_size": torch.tensor(1),
                "device": None,
                "items": {},
            },
            "epoch_loop._results": {
                "training": True,
                "_minimize": None,
                "_batch_size": torch.tensor(1),
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
            "dataloader_progress": {
                "total": {"ready": 0, "started": None, "processed": None, "completed": 0},
                "current": {"ready": 0, "started": None, "processed": None, "completed": 0},
            },
            "epoch_loop.state_dict": {},
            "epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            },
            "_results": {
                "training": False,
                "_minimize": None,
                "_batch_size": torch.tensor(1),
                "device": None,
                "items": {},
            },
        },
        "test_loop": {
            "state_dict": {},
            "dataloader_progress": {
                "total": {"ready": 0, "started": None, "processed": None, "completed": 0},
                "current": {"ready": 0, "started": None, "processed": None, "completed": 0},
            },
            "epoch_loop.state_dict": {},
            "epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            },
            "_results": {
                "training": False,
                "_minimize": None,
                "_batch_size": torch.tensor(1),
                "device": None,
                "items": {},
            },
        },
        "predict_loop": {
            "state_dict": {},
            "dataloader_progress": {
                "total": {"ready": 0, "started": None, "processed": None, "completed": 0},
                "current": {"ready": 0, "started": None, "processed": None, "completed": 0},
            },
            "epoch_loop.state_dict": {},
            "epoch_loop.batch_progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
            },
        },
    }
    # yapf: enable
    assert state_dict == expected


@mock.patch.dict(os.environ, {"PL_FAULT_TOLERANT_TRAINING": "1"})
def test_loops_state_dict_structure_short_training(tmpdir):
    # todo (tchaton) Bug with val_datalaoders being shared across validate_loop and val_loop.
    # States are being lost
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=3, limit_val_batches=5)
    trainer.fit(model)
    trainer.limit_val_batches = 11
    trainer.validate(model)

    state_dict = {
        "fit_loop": trainer.fit_loop.state_dict(),
        "validate_loop": trainer.validate_loop.state_dict(),
    }

    # yapf: disable
    expected = {
        "fit_loop": {
            "state_dict": {
                "dataloader": {
                    "num_workers": 0,
                    "previous_worker": None,
                    0: {"current_iteration": 3},
                }
            },
            "epoch_loop.state_dict": {},
            "epoch_loop.progress": {
                "total": {"ready": 1, "started": 1, "processed": 1, "completed": 1},
                "current": {"ready": 1, "started": 1, "processed": 1, "completed": 1},
                "should_check_val": True,
            },
            "epoch_loop.batch_loop.state_dict": {},
            "epoch_loop.batch_loop.progress": {
                "total": {"ready": 3, "started": 3, "processed": 3, "completed": 3},
                "current": {"ready": 3, "started": 3, "processed": 3, "completed": 3},
            },
            "epoch_loop.batch_loop.optim_progress": {
                "optimizer_idx": 0,
                "optimizer": {
                    "step": {
                        "total": {
                            "ready": 3,
                            "started": 3,
                            "processed": None,
                            "completed": 3,
                        },
                        "current": {
                            "ready": 1,
                            "started": 1,
                            "processed": None,
                            "completed": 1,
                        },
                    },
                    "zero_grad": {
                        "total": {
                            "ready": 3,
                            "started": 3,
                            "processed": None,
                            "completed": 3,
                        },
                        "current": {
                            "ready": 1,
                            "started": 1,
                            "processed": None,
                            "completed": 1,
                        },
                    },
                },
                "scheduler": {
                    "total": {
                        "ready": 1,
                        "started": None,
                        "processed": None,
                        "completed": 1,
                    },
                    "current": {
                        "ready": 1,
                        "started": None,
                        "processed": None,
                        "completed": 1,
                    },
                },
            },
            "epoch_loop.val_loop.state_dict": {
                "dataloader": {
                    "num_workers": 0,
                    "previous_worker": None,
                    # todo bug: should be 5.
                    0: {"current_iteration": 11},
                }
            },
            "epoch_loop.val_loop.progress": {
                "total": {"ready": 2, "started": 2, "processed": 2, "completed": 2},
                "current": {"ready": 1, "started": 1, "processed": 1, "completed": 1},
                "dataloader_idx": 0,
            },
            "epoch_loop.val_loop.epoch_loop.state_dict": {},
            "epoch_loop.val_loop.epoch_loop.progress": {
                # sanity checking has been counted with val loop
                "total": {"ready": 7, "started": 7, "processed": 7, "completed": 7},
                "current": {"ready": 5, "started": 5, "processed": 5, "completed": 5},
            },
        },
        "validate_loop": {
            "state_dict": {
                "dataloader": {
                    "num_workers": 0,
                    "previous_worker": None,
                    0: {"current_iteration": 11},
                }
            },
            "progress": {
                "total": {"ready": 1, "started": 1, "processed": 1, "completed": 1},
                "current": {"ready": 1, "started": 1, "processed": 1, "completed": 1},
                "dataloader_idx": 0,
            },
            "epoch_loop.state_dict": {},
            "epoch_loop.progress": {
                "total": {"ready": 11, "started": 11, "processed": 11, "completed": 11},
                "current": {"ready": 11, "started": 11, "processed": 11, "completed": 11},
            },
        },
    }
    # yapf: enable
    assert state_dict == expected
