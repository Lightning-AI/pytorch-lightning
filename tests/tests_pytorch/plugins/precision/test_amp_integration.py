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
from unittest.mock import Mock

import torch
from lightning.fabric import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins.precision import MixedPrecision

from tests_pytorch.helpers.runif import RunIf


class FusedOptimizerParityModel(BoringModel):
    def __init__(self, fused=False):
        super().__init__()
        self.fused = fused

    def configure_optimizers(self):
        assert isinstance(self.trainer.precision_plugin.scaler, torch.cuda.amp.GradScaler)
        return torch.optim.Adam(self.parameters(), lr=1.0, fused=self.fused)


@RunIf(min_cuda_gpus=1)
def test_amp_fused_optimizer_parity(tmp_path):
    def run(fused=False):
        seed_everything(1234)
        model = FusedOptimizerParityModel(fused)
        trainer = Trainer(
            default_root_dir=tmp_path,
            accelerator="cuda",
            devices=1,
            precision="16-mixed",
            max_steps=5,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model)
        return model.parameters()

    params = run(fused=False)
    params_fused = run(fused=True)

    # Both the regular and the fused version of Adam produce the same losses and model weights
    for p, q in zip(params, params_fused):
        torch.testing.assert_close(p, q)


@RunIf(min_cuda_gpus=1)
def test_skip_training_step_with_grad_scaler():
    """Test that the grad scaler gets skipped when skipping a training step."""

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            if batch_idx % 2:
                return None  # skipping the backward should skip the grad scaler too
            return super().training_step(batch, batch_idx)

    trainer = Trainer(
        accelerator="cuda",
        devices=1,
        precision="16-mixed",
        barebones=True,
        max_steps=5,
        gradient_clip_val=0.5,
    )
    assert isinstance(trainer.precision_plugin, MixedPrecision)
    assert trainer.precision_plugin.scaler is not None
    trainer.precision_plugin.scaler = Mock(wraps=trainer.precision_plugin.scaler)
    model = TestModel()
    trainer.fit(model)
    assert trainer.precision_plugin.scaler.unscale_.call_count == 3
    assert trainer.precision_plugin.scaler.step.call_count == 3
    assert trainer.precision_plugin.scaler.update.call_count == 3
