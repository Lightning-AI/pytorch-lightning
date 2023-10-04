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
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from torch.utils.data import Dataset


class RandomDatasetA(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return torch.zeros(1)

    def __len__(self):
        return self.len


class RandomDatasetB(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return torch.ones(1)

    def __len__(self):
        return self.len


@pytest.mark.parametrize("seq_type", [tuple, list])
def test_multiple_eval_dataloaders_seq(tmpdir, seq_type):
    class TestModel(BoringModel):
        def validation_step(self, batch, batch_idx, dataloader_idx):
            if dataloader_idx == 0:
                assert batch.sum() == 0
            elif dataloader_idx == 1:
                assert batch.sum() == 11
            else:
                raise Exception("should only have two dataloaders")

        def val_dataloader(self):
            dl1 = torch.utils.data.DataLoader(RandomDatasetA(32, 64), batch_size=11)
            dl2 = torch.utils.data.DataLoader(RandomDatasetB(32, 64), batch_size=11)
            return seq_type((dl1, dl2))

    model = TestModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        enable_model_summary=False,
    )

    trainer.fit(model)
