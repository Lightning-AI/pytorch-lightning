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

import lightning.pytorch as pl
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.model_summary import DeepSpeedSummary

from tests_pytorch.helpers.runif import RunIf


@RunIf(min_cuda_gpus=2, deepspeed=True, standalone=True)
def test_deepspeed_summary(tmpdir):
    """Test to ensure that the summary contains the correct values when stage 3 is enabled and that the trainer enables
    the `DeepSpeedSummary` when DeepSpeed is used."""

    model = BoringModel()
    total_parameters = sum(x.numel() for x in model.parameters())

    class TestCallback(Callback):
        def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            model_summary = DeepSpeedSummary(pl_module, max_depth=1)
            assert model_summary.total_parameters == total_parameters
            assert model_summary.trainable_parameters == total_parameters

            # check the additional params per device
            summary_data = model_summary._get_summary_data()
            params_per_device = summary_data[-1][-1]
            assert int(params_per_device[0]) == (model_summary.total_parameters // 2)

    trainer = Trainer(
        strategy=DeepSpeedStrategy(stage=3),
        default_root_dir=tmpdir,
        accelerator="gpu",
        fast_dev_run=True,
        devices=2,
        precision="16-mixed",
        enable_model_summary=True,
        callbacks=[TestCallback()],
    )

    trainer.fit(model)
