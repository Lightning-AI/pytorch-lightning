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
import pytest
from tests.base.boring_model import BoringModel, RandomDataset
from pytorch_lightning import Trainer


def test_overfit_multiple_val_loaders(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):

        def validation_step(self, batch, batch_idx, dataloader_idx):
            output = self.layer(batch[0])
            loss = self.loss(batch, output)
            return {"x": loss}

        def validation_epoch_end(self, outputs) -> None:
            pass

        def val_dataloader(self):
            dl1 = torch.utils.data.DataLoader(RandomDataset(32, 64))
            dl2 = torch.utils.data.DataLoader(RandomDataset(32, 64))
            return [dl1, dl2]

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        overfit_batches=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)


@pytest.mark.parametrize('overfit', [1, 2, 0.1, 0.25, 1.0])
def test_overfit_basic(tmpdir, overfit):
    """
    Tests that only training_step can be used
    """

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        overfit_batches=overfit,
        weights_summary=None,
    )

    trainer.fit(model)
