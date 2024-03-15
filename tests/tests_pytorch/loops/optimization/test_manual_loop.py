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
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops.optimization.manual import ManualResult
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def test_manual_result():
    training_step_output = {"loss": torch.tensor(25.0, requires_grad=True), "something": "jiraffe"}
    result = ManualResult.from_training_step_output(training_step_output)
    asdict = result.asdict()
    assert not asdict["loss"].requires_grad
    assert asdict["loss"] == 25
    assert result.extra == asdict


def test_warning_invalid_trainstep_output(tmp_path):
    class InvalidTrainStepModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        def training_step(self, batch, batch_idx):
            return 5

    model = InvalidTrainStepModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=1)

    with pytest.raises(MisconfigurationException, match="return a Tensor or have no return"):
        trainer.fit(model)
