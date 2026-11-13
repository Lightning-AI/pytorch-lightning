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
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.tuner import Tuner


class _IterableLRData(IterableDataset):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        yield from (torch.tensor([1.0 + i / 10]) for i in range(self.size))


class _TailLRModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.lr = 0.01
        self.optimizer_lrs = []

    def training_step(self, batch, batch_idx):
        return (self.weight * batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)

    def on_before_optimizer_step(self, optimizer):
        self.optimizer_lrs.append(optimizer.param_groups[0]["lr"])


@pytest.mark.parametrize(("batches", "accumulation"), [(1, 4), (3, 2), (5, 3), (4, 2)])
@pytest.mark.parametrize("mode", ["linear", "exponential"])
@pytest.mark.parametrize("iterable", [False, True])
def test_lr_finder_records_final_accumulation_step(tmp_path, batches, accumulation, mode, iterable):
    dataset = (
        _IterableLRData(batches) if iterable else TensorDataset(torch.arange(batches).float().unsqueeze(1) / 10 + 1)
    )
    dataloader = DataLoader(dataset, batch_size=1)
    model = _TailLRModel()
    initial = deepcopy(model.state_dict())
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=20,
        accumulate_grad_batches=accumulation,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    result = Tuner(trainer).lr_find(
        model,
        train_dataloaders=dataloader,
        min_lr=1e-5,
        max_lr=1e-3,
        num_training=6,
        mode=mode,
        early_stop_threshold=None,
        update_attr=False,
    )
    assert result is not None
    assert len(model.optimizer_lrs) == 6
    assert result.results["lr"] == pytest.approx(model.optimizer_lrs)
    assert len(result.results["loss"]) == 6
    assert torch.isfinite(torch.tensor(result.results["loss"])).all()
    assert trainer.global_step == 0
    torch.testing.assert_close(model.state_dict(), initial, rtol=0, atol=0)

    trainer.fit_loop.max_epochs = 1
    trainer.fit(model, train_dataloaders=dataloader)
    expected_steps = (batches + accumulation - 1) // accumulation
    assert trainer.global_step == expected_steps
    assert model.weight.item() != initial["weight"].item()
