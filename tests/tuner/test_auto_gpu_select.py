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
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base.boring_model import BoringModel
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.auto_gpu_select import pick_multiple_gpus


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="test requires a number of GPU machine greater than 1"
)
@pytest.mark.parametrize(
    ["auto_select_gpus", "gpus", "expected_error"],
    [
        pytest.param(True, 0, MisconfigurationException),
        pytest.param(True, -1, None),
        pytest.param(False, 0, None),
        pytest.param(False, -1, None),
    ],
)
def test_combination_gpus_options(auto_select_gpus, gpus, expected_error):
    model = BoringModel()

    if expected_error:
        with pytest.raises(expected_error):
            trainer = Trainer(auto_select_gpus=auto_select_gpus, gpus=gpus)
    else:
        trainer = Trainer(auto_select_gpus=auto_select_gpus, gpus=gpus)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="test requires a number of GPU machine greater than 1"
)
@pytest.mark.parametrize(
    ["nb", "expected_gpu_idxs"],
    [
        pytest.param(0, []),
        pytest.param(-1, [i for i in range(torch.cuda.device_count())]),
        pytest.param(1, [0]),
    ],
)
def test_pick_multiple_gpus(nb, expected_gpu_idxs):
    assert expected_gpu_idxs == pick_multiple_gpus(nb)