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

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.strategies import BaguaStrategy
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


class TestModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)


@RunIf(skip_windows=True, bagua=True, min_gpus=2, standalone=True)
def test_bagua_algorithm(tmpdir):
    model = TestModel()
    bagua_strategy = BaguaStrategy(algorithm="gradient_allreduce")
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=2,
    )
    trainer.fit(model)

    for param in model.parameters():
        assert torch.norm(param) < 3


@RunIf(bagua=False, min_gpus=1)
def test_bagua_not_available():

    with pytest.raises(MisconfigurationException, match="you must have `Bagua` installed"):
        trainer = Trainer(strategy="bagua", gpus=1)
