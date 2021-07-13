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

from pytorch_lightning.loops import FitLoop
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_loops_state_dict():
    fit_loop = FitLoop()
    with pytest.raises(MisconfigurationException, match="Loop FitLoop should be connected to a"):
        fit_loop.connect(object())  # noqa

    fit_loop.connect(Trainer())
    state_dict = fit_loop.state_dict()
    new_fit_loop = FitLoop()
    new_fit_loop.load_state_dict(state_dict)
    assert fit_loop.state_dict() == new_fit_loop.state_dict()


def test_loops_state_dict_structure():
    trainer = Trainer()
    # structure saved by the checkpoint connector
    state_dict = {
        "fit_loop": trainer.fit_loop.state_dict(),
        "validate_loop": trainer.validate_loop.state_dict(),
        "test_loop": trainer.test_loop.state_dict(),
        "predict_loop": trainer.predict_loop.state_dict(),
    }
    # todo (tchaton) Update this once new progress as been added.
    # yapf: disable
    expected = {
        "fit_loop": {
            "epoch_loop": {
                "batch_loop": {
                    "state_dict": {},
                    "progress": {
                        "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                        "current": {
                            "ready": 0,
                            "started": 0,
                            "processed": 0,
                            "completed": 0,
                        },
                    },
                    "optim_progress": {
                        "optimizer": {
                            "step": {
                                "total": {
                                    "ready": 0,
                                    "started": 0,
                                    "processed": None,
                                    "completed": 0,
                                },
                                "current": {
                                    "ready": 0,
                                    "started": 0,
                                    "processed": None,
                                    "completed": 0,
                                },
                            },
                            "zero_grad": {
                                "total": {
                                    "ready": 0,
                                    "started": 0,
                                    "processed": None,
                                    "completed": 0,
                                },
                                "current": {
                                    "ready": 0,
                                    "started": 0,
                                    "processed": None,
                                    "completed": 0,
                                },
                            },
                        },
                        "scheduler": {
                            "total": {
                                "ready": 0,
                                "started": None,
                                "processed": None,
                                "completed": 0,
                            },
                            "current": {
                                "ready": 0,
                                "started": None,
                                "processed": None,
                                "completed": 0,
                            },
                        },
                    },
                },
                "val_loop": {
                    "state_dict": {},
                    "progress": {
                        "epoch": {
                            "total": {
                                "ready": 0,
                                "started": 0,
                                "processed": 0,
                                "completed": 0,
                            },
                            "current": {
                                "ready": 0,
                                "started": 0,
                                "processed": 0,
                                "completed": 0,
                            },
                            "batch": {
                                "total": {
                                    "ready": 0,
                                    "started": 0,
                                    "processed": 0,
                                    "completed": 0,
                                },
                                "current": {
                                    "ready": 0,
                                    "started": 0,
                                    "processed": 0,
                                    "completed": 0,
                                },
                            },
                        }
                    },
                    "epoch_loop.state_dict": {},
                    "epoch_loop.progress": {
                        "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                        "current": {
                            "ready": 0,
                            "started": 0,
                            "processed": 0,
                            "completed": 0,
                        },
                        "batch": {
                            "total": {
                                "ready": 0,
                                "started": 0,
                                "processed": 0,
                                "completed": 0,
                            },
                            "current": {
                                "ready": 0,
                                "started": 0,
                                "processed": 0,
                                "completed": 0,
                            },
                        },
                    },
                },
            }
        },
        "validate_loop": {
            "state_dict": {},
            "progress": {
                "epoch": {
                    "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                    "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                    "batch": {
                        "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                        "current": {
                            "ready": 0,
                            "started": 0,
                            "processed": 0,
                            "completed": 0,
                        },
                    },
                }
            },
            "epoch_loop.state_dict": {},
            "epoch_loop.progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "batch": {
                    "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                    "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                },
            },
        },
        "test_loop": {
            "state_dict": {},
            "progress": {
                "epoch": {
                    "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                    "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                    "batch": {
                        "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                        "current": {
                            "ready": 0,
                            "started": 0,
                            "processed": 0,
                            "completed": 0,
                        },
                    },
                }
            },
            "epoch_loop.state_dict": {},
            "epoch_loop.progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "batch": {
                    "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                    "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                },
            },
        },
        "predict_loop": {
            "state_dict": {},
            "progress": {
                "epoch": {
                    "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                    "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                    "batch": {
                        "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                        "current": {
                            "ready": 0,
                            "started": 0,
                            "processed": 0,
                            "completed": 0,
                        },
                    },
                }
            },
            "epoch_loop.state_dict": {},
            "epoch_loop.progress": {
                "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                "batch": {
                    "total": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                    "current": {"ready": 0, "started": 0, "processed": 0, "completed": 0},
                },
            },
        },
    }
    # yapf: enable
    assert state_dict == expected
