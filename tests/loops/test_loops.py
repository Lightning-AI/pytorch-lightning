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
from unittest.mock import ANY

import pytest

from pytorch_lightning.loops import FitLoop
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_loops_state_dict_structure():

    fit_loop = FitLoop()
    with pytest.raises(
        MisconfigurationException, match="The Trainer should be connected to loop to retrieve the state_dict."
    ):
        state_dict = fit_loop.state_dict()
    with pytest.raises(
        MisconfigurationException,
        match="Loop FitLoop should be connected to a :class:`~pytorch_lightning.Trainer` instance."
    ):
        fit_loop.connect(object())
    fit_loop.connect(Trainer())
    state_dict = fit_loop.state_dict()
    expected = {'epoch_loop': {'batch_loop': ANY, 'val_loop': ANY}}
    assert state_dict == expected

    with pytest.raises(NotImplementedError):
        fit_loop.load_state_dict(state_dict)
