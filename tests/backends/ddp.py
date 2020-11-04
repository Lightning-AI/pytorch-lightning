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
import torch
import pytest
import collections
from tests.base.boring_model import BoringModel, RandomDataset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import APEX_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

def test_automatic_optimization_false_ddp(args):
    """
    This test verify that in `automatic_optimization` we don't add gradient if the user return loss.
    """

    class ExtendedModel(BoringModel):

        count = 0
        called = collections.defaultdict(int)

        @property
        def should_update(self):
            return self.count % 2 == 0

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            self.called["on_train_batch_start"] += 1
            self.weight_before = self.layer.weight.clone()

        def training_step(self, batch, batch_idx):
            self.called["training_step"] += 1
            opt = self.optimizers()
            output = self.layer(batch)
            loss = self.loss(batch, output)
            print(self.device, self.count)
            if self.should_update:
                loss = self.trainer.scaler.scale(loss)

                weight_before = self.layer.weight.clone()
                self.manual_backward(loss, opt)

                print(self.device, torch.sum(self.layer.weight.grad))

                assert torch.sum(self.layer.weight.grad) != 0

                #self.trainer.scaler.unscale_(opt)

                print(self.device, self.layer.weight.grad)

                opt.step()

                after_before = self.layer.weight.clone()

                assert not torch.equal(weight_before, after_before)

                opt.zero_grad()

            # the loss should be ignored
            return loss

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.called["on_train_batch_end"] += 1
            after_before = self.layer.weight.clone()
            if self.should_update:
                assert not torch.equal(self.weight_before, after_before)
            else:
                #if not torch.isnan(after_before) or
                assert torch.equal(self.weight_before, after_before)
            assert torch.sum(self.layer.weight.grad) == 0
            self.count += 1

    model = ExtendedModel()
    model.training_step_end = None
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=args.tmpdir,
        limit_train_batches=10,
        limit_test_batches=0,
        limit_val_batches=0,
        automatic_optimization=False,
        precision=16,
        amp_backend='native',
        accelerator="ddp",
        gpus=2,
    )
    trainer.fit(model)

    assert model.called["training_step"] == 10
    assert model.called["on_train_batch_start"] == 10
    assert model.called["on_train_batch_end"] == 10

    return {'status': 'complete', 'method': args.func_name, 'result': None}
