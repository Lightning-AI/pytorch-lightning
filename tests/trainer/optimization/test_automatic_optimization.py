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

import pytest
import torch
import torch.nn.functional as F

from pytorch_lightning import Trainer
from tests.base.boring_model import BoringModel


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
@pytest.mark.parametrize('accumulate_grad_batches', [1, 2])
@pytest.mark.parametrize('invalid_loss_strategy', ["normal", "skip_if_any", "never_skip"])
def test_automatic_optimization_with_nan_loss_and_ddp(tmpdir, accumulate_grad_batches, invalid_loss_strategy):
    """
    Tests that training doesn't hang with returning nan loss
    """
    class TestModel(BoringModel):
        def __init__(self, invalid_loss_strategy):
            super().__init__()
            self._invalid_loss_strategy = invalid_loss_strategy

        def training_step(self, batch, batch_idx):
            local_rank = os.getenv("LOCAL_RANK")
            rank = str(int(batch_idx % 2 == 0))
            output = super().training_step(batch, batch_idx)["loss"]
            if local_rank == rank:
                if batch_idx in [0, 1, 8, 9] and self.invalid_loss_strategy != "never_skip":
                    output = None
                elif batch_idx in [2, 3, 10, 11]:
                    output = torch.tensor(float('NaN'), device=self.device)
            return output

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer

        @property
        def invalid_loss_strategy(self):
            return self._invalid_loss_strategy

        def on_train_epoch_end(self, *_) -> None:
            clone_weight = self.layer.weight.data.clone()
            weight = self.layer.weight.data
            self.trainer.accelerator_backend.sync_tensor(weight)
            torch.equal(weight / 2., clone_weight)

    model = TestModel(invalid_loss_strategy)
    model.val_dataloader = None
    model.training_epoch_end = None

    try:
        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=12,
            limit_val_batches=2,
            max_epochs=1,
            log_every_n_steps=1,
            weights_summary=None,
            gpus=2,
            accelerator="ddp",
            accumulate_grad_batches=accumulate_grad_batches,
        )
        trainer.fit(model)
    except Exception as e:
        msg = "LightningModule `invalid_loss_strategy` property should be within"
        if msg in str(e) and invalid_loss_strategy == "normal":
            pass
        else:
            raise Exception(str(e))
