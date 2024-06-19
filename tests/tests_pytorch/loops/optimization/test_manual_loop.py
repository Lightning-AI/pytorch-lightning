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
from typing import Any

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.loops.optimization.manual import ManualResult
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT


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


# DO NOT SUBMIT:
# move to tests/tests_pytorch/trainer/optimization/test_manual_optimization.py
class MultiOptimizerModel(BoringModel):
    def __init__(self) -> None:
        # This initializes `BoringModel`'s parent, but skips `BoringModel`
        # itself (as we don't need its `layer`)
        super(BoringModel, self).__init__()
        self.layer_1 = torch.nn.Linear(32, 4)
        self.layer_2 = torch.nn.Linear(4, 2)
        self.layer_3 = torch.nn.Linear(2, 1)
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(torch.relu(self.layer_2(torch.relu(self.layer_1(x)))))

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        # self.optimizers().zero_grad()
        for i, opt in enumerate(self.optimizers()):
            opt.zero_grad()
            # if i+1 < len(self.optimizers()):
            #     opt._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
            #     opt._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")

        loss = self.step(batch)
        self.manual_backward(loss)

        # self.optimizers().step()
        for opt in self.optimizers():
            opt.step()
        return {"loss": loss}

    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
    #     for sch in self.lr_schedulers():
    #         sch.step()

    def configure_optimizers(self):
        opt1 = torch.optim.SGD(self.layer_1.parameters(), lr=0.1)
        opt2 = torch.optim.SGD(self.layer_2.parameters(), lr=0.2)
        opt3 = torch.optim.SGD(self.layer_3.parameters(), lr=0.3)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # increments_step
        # test single dict with multi schedulers
        return (
            {"optimizer": [opt1, opt2], "should_increment": [True, True]},
            {"optimizer": [opt3], "should_increment": False},
        )


# DO NOT SUBMIT: Parameterize to test default=no extra, explicit=is extra steps
def test_multiple_optimizers(tmpdir):
    # Tests the global number of steps interaction with multiple optimizers,
    # as well as that multiple optimizers are still all run.
    model = MultiOptimizerModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=1,
    )
    trainer.fit(model)
    print(trainer.global_step)
    assert trainer.global_step == 8
