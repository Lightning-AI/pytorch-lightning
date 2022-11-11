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
import torch

from lightning_lite import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel
from tests_pytorch.helpers.runif import RunIf


class FusedOptimizerParityModel(BoringModel):
    def __init__(self, fused=False):
        super().__init__()
        self.fused = fused

    def configure_optimizers(self):
        assert isinstance(self.trainer.precision_plugin.scaler, torch.cuda.amp.GradScaler)
        return torch.optim.Adam(self.parameters(), lr=1.0, fused=self.fused)


@RunIf(min_torch="1.13", min_cuda_gpus=1)
def test_native_mixed_precision_fused_optimizer_parity(tmpdir):
    def run(fused=False):
        seed_everything(1234)
        model = FusedOptimizerParityModel(fused)
        trainer = Trainer(
            default_root_dir=tmpdir,
            accelerator="cuda",
            devices=1,
            precision=16,
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
