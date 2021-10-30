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
from torch import nn

from pytorch_lightning.core.system import LightningSystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_lightning_system(tmpdir):
    class System(LightningSystem):
        def ___init__(self):
            self.model = nn.Linear(1, 1)

    system = System()

    system.model = nn.Linear(1, 1)

    with pytest.raises(
        MisconfigurationException,
        match="A `LightningSystem` supports' only a single nn.Module expects `torchmetrics.Metric`",
    ):
        setattr(system, "model_1", nn.Linear(1, 1))

    with pytest.raises(
        MisconfigurationException,
        match="A `LightningSystem` doesn't support parameters.",
    ):
        setattr(system, "weight", nn.Parameter(torch.Tensor(1, 1)))
