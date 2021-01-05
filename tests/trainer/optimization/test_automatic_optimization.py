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
import collections
import os
from unittest import mock
from unittest.mock import ANY, call, patch

import pytest
import torch
import torch.distributed as torch_distrib
import torch.nn.functional as F

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import APEX_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base.boring_model import BoringModel


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not os.getenv("PL_RUNNING_SPECIAL_TESTS", '0') == '1',
                    reason="test should be run outside of pytest")
@pytest.mark.parametrize('accumulate_grad_batches', [1])
def test_automatic_optimization_with_nan_loss_and_ddp(tmpdir, accumulate_grad_batches):
    """
    Tests that training doesn't hang with returning nan loss
    """
    class TestModel(BoringModel):
        """
        def training_step(self, batch, batch_idx):
            if os.getenv("LOCAL_RANK") == str(batch_idx % 2 == 0):
                return torch.tensor(float('NaN'), device=self.device)
            return super().training_step(batch, batch_idx)
        """
        def training_step(self, batch, batch_idx):
            if os.getenv("LOCAL_RANK") == str(batch_idx % 2 == 0) or batch_idx == 12:
                if batch_idx in [1, 2, 5, 7, 9, 11, 12]:
                    return torch.tensor(float('NaN'), device=self.device)
            return super().training_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer

        @property
        def is_loss_possibly_nan(self) -> bool:
            return True

        def on_train_epoch_end(self, *_) -> None:
            clone_weight = self.layer.weight.data.clone()
            weight = self.layer.weight.data
            self.trainer.accelerator_backend.sync_tensor(weight)
            torch.equal(weight / 2., clone_weight)

    model = TestModel()
    model.val_dataloader = None
    model.training_epoch_end = None

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
