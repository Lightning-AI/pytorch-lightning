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

import pytest
import torch
from torch import nn

from pytorch_lightning.core.system import from_checkpoint, LightningSystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def test_lightning_system(tmpdir):

    system = LightningSystem()
    assert not system.has_module
    assert system.module_name is None

    linear = nn.Linear(in_features=1, out_features=1)
    system.model = linear
    assert system._args == ()
    assert system._kwargs == dict(in_features=1, out_features=1)
    assert system.has_module
    assert system.module_name == "model"

    with pytest.raises(
        MisconfigurationException,
        match="A `LightningSystem` supports only a single nn.Module expects `torchmetrics.Metric`",
    ):
        setattr(system, "model_1", nn.Linear(in_features=1, out_features=1))

    with pytest.raises(
        MisconfigurationException,
        match="A `LightningSystem` doesn't support parameters.",
    ):
        setattr(system, "weight", nn.Parameter(torch.Tensor(1, 1)))

    state_dict = deepcopy(system.state_dict())
    assert state_dict["weight"] == linear.state_dict()["weight"]

    system.load_state_dict(nn.Linear(in_features=1, out_features=1).state_dict())
    assert system.state_dict()["weight"] != state_dict["weight"]

    class DummyModel(nn.Module):
        def __init__(self, name: str, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

    class System(LightningSystem):
        def __init__(self, name: str, in_features, out_features, random):
            super().__init__()
            self.save_hyperparameters()
            self.module = DummyModel(name, in_features, out_features + 1)

    system = System("dummy", 1, 1, "random")

    p = os.path.join(tmpdir, "model.pt")
    torch.save(state_dict, p)
    linear = from_checkpoint(nn.Linear, p)
    assert linear.state_dict()["weight"] == state_dict["weight"]
