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
import os
import torch
import torch.nn as nn


from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from tests.base.boring_model import BoringModel, RandomDictDataset, RandomDictStringDataset


def test_lightning_optimizer(tmpdir):

    class TestModel(BoringModel):
        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            # optimizer = LightningOptimizer(self.trainer, optimizer)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

        def optimizer_step(self, *args, epoch = None, batch_idx = None, optimizer = None,
                           optimizer_idx = None, optimizer_closure = None, on_tpu = None, using_native_amp = None,
                           using_lbfgs = None, **kwargs) -> None:
            optimizer.step(optimizer_closure)


    # model
    model = TestModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)
