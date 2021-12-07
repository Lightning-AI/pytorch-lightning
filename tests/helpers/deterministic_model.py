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
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.core.lightning import LightningModule


class DeterministicModel(LightningModule):
    def __init__(self, weights=None):
        super().__init__()

        self.training_step_called = False
        self.training_step_end_called = False
        self.training_epoch_end_called = False

        self.validation_step_called = False
        self.validation_step_end_called = False
        self.validation_epoch_end_called = False

        self.assert_backward = True

        self.l1 = nn.Linear(2, 3, bias=False)
        if weights is None:
            weights = torch.tensor([[4, 3, 5], [10, 11, 13]]).float()
            p = torch.nn.Parameter(weights, requires_grad=True)
            self.l1.weight = p

    def forward(self, x):
        return self.l1(x)

    def step(self, batch, batch_idx):
        x = batch
        bs = x.size(0)
        y_hat = self.l1(x)

        test_hat = y_hat.cpu().detach()
        assert torch.all(test_hat[:, 0] == 15.0)
        assert torch.all(test_hat[:, 1] == 42.0)
        out = y_hat.sum()
        assert out == (42.0 * bs) + (15.0 * bs)

        return out

    def count_num_graphs(self, result, num_graphs=0):
        for k, v in result.items():
            if isinstance(v, torch.Tensor) and v.grad_fn is not None:
                num_graphs += 1
            if isinstance(v, dict):
                num_graphs += self.count_num_graphs(v)

        return num_graphs

    def validation_step_end(self, val_step_output):
        assert len(val_step_output) == 3
        assert val_step_output["val_loss"] == 171
        assert val_step_output["log"]["log_acc1"] >= 12
        assert val_step_output["progress_bar"]["pbar_acc1"] == 17
        self.validation_step_end_called = True

        val_step_output["val_step_end"] = torch.tensor(1802)

        return val_step_output

    def validation_epoch_end(self, outputs):
        assert len(outputs) == self.trainer.num_val_batches[0]

        for i, out in enumerate(outputs):
            assert out["log"]["log_acc1"] >= 12 + i

        self.validation_epoch_end_called = True

        result = outputs[-1]
        result["val_epoch_end"] = torch.tensor(1233)
        return result

    # -----------------------------
    # DATA
    # -----------------------------
    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=3, shuffle=False)

    def val_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=3, shuffle=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0)

    def configure_optimizers__lr_on_plateau_epoch(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = {"scheduler": lr_scheduler, "interval": "epoch", "monitor": "epoch_end_log_1"}
        return [optimizer], [scheduler]

    def configure_optimizers__lr_on_plateau_step(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "monitor": "pbar_acc1"}
        return [optimizer], [scheduler]

    def backward(self, loss, optimizer, optimizer_idx):
        if self.assert_backward:
            if self.trainer.precision == 16:
                assert loss > 171 * 1000
            else:
                assert loss == 171.0

        super().backward(loss, optimizer, optimizer_idx)


class DummyDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, idx):
        return torch.tensor([0.5, 1.0, 2.0])
