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


def test_loops_state_dict_structure():
    fit_loop = FitLoop()
    with pytest.raises(MisconfigurationException, match="Loop FitLoop should be connected to a"):
        fit_loop.connect(object())  # noqa

    fit_loop.connect(Trainer())
    state_dict = fit_loop.state_dict()
    new_fit_loop = FitLoop()
    new_fit_loop.load_state_dict(state_dict)
    assert fit_loop.state_dict() == new_fit_loop.state_dict()


def test_loops_state_dict_structure_with_trainer():
    trainer = Trainer()
    state_dict = trainer.get_loops_state_dict()
    expected = {
        "fit_loop": {
            'epoch_loop': {
                'batch_loop': {},
                'val_loop': {},
            }
        },
        "validate_loop": {},
        "test_loop": {},
    }
    assert state_dict == expected
