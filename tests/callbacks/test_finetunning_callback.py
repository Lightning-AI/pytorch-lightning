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
from torch import nn

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import BackboneFinetuning
from tests.base import BoringModel


def test_finetunning_callback(tmpdir):
    """Test finetunning callbacks works as expected"""

    seed_everything(42)

    class FinetunningBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(32, 32, bias=False), nn.BatchNorm1d(32), nn.ReLU())
            self.layer = torch.nn.Linear(32, 2)

        def forward(self, x):
            x = self.backbone(x)
            return self.layer(x)

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
            return [optimizer], [lr_scheduler]

    class TestCallback(BackboneFinetuning):

        def on_train_epoch_end(self, trainer, pl_module, outputs):
            epoch = trainer.current_epoch
            if self.unfreeze_backbone_at_epoch <= epoch:
                optimizer = trainer.optimizers[0]
                current_lr = optimizer.param_groups[0]['lr']
                backbone_lr = self.previous_backbone_lr
                if epoch < 6:
                    assert backbone_lr <= current_lr
                else:
                    assert backbone_lr == current_lr

    model = FinetunningBoringModel()
    callback = TestCallback(unfreeze_backbone_at_epoch=3, verbose=False)

    trainer = Trainer(
        limit_train_batches=1,
        default_root_dir=tmpdir,
        callbacks=[callback],
        max_epochs=8,
    )
    trainer.fit(model)
