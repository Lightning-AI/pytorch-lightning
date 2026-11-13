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
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(), (1,)])
def test_lr_monitor_snapshots_mutable_tensor_lr(dtype, shape):
    monitor = LearningRateMonitor()
    monitor.lrs["lr"] = []
    lr = torch.full(shape, 0.125, dtype=dtype)
    group = {"lr": lr}
    first = monitor._extract_lr(group, "lr")
    lr.mul_(0.5)
    second = monitor._extract_lr(group, "lr")
    lr.fill_(0.001)
    assert monitor.lrs["lr"] == pytest.approx([0.125, 0.0625])
    assert first["lr"] == pytest.approx(0.125)
    assert second["lr"] == pytest.approx(0.0625)
    assert all(isinstance(value, float) for value in monitor.lrs["lr"])
    assert group["lr"] is lr


class _TensorLRModel(LightningModule):
    def __init__(self, optimizer_type, tensor_lr, dtype, interval):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))
        self.optimizer_type = optimizer_type
        self.initial_lr = torch.tensor(0.125, dtype=dtype) if tensor_lr else 0.125
        self.interval = interval
        self.step_lrs = []

    def configure_optimizers(self):
        optimizer = self.optimizer_type(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": self.interval}}

    def training_step(self, batch, batch_idx):
        return (self.weight * batch[0]).square().mean()

    def on_before_optimizer_step(self, optimizer):
        self.step_lrs.append(float(optimizer.param_groups[0]["lr"]))


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.AdamW])
@pytest.mark.parametrize("tensor_lr", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("interval", ["step", "epoch"])
def test_lr_monitor_tensor_history_matches_training(tmp_path, optimizer_type, tensor_lr, dtype, interval):
    model = _TensorLRModel(optimizer_type, tensor_lr, dtype, interval)
    monitor = LearningRateMonitor()
    trainer = Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        callbacks=[monitor],
        logger=CSVLogger(tmp_path),
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    data = torch.tensor([1.0, 2.0], dtype=dtype)
    trainer.fit(model, train_dataloaders=DataLoader(TensorDataset(data), batch_size=1))
    history = monitor.lrs[f"lr-{optimizer_type.__name__}"]
    expected = model.step_lrs if interval == "step" else model.step_lrs[::2]
    assert history == pytest.approx(expected)
    assert len(set(history)) == len(history)

    reference = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))
    reference_optimizer = optimizer_type([reference], lr=0.125)
    scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=1, gamma=0.5)
    for _ in range(3):
        for value in data:
            reference_optimizer.zero_grad()
            (reference * value).square().backward()
            reference_optimizer.step()
            if interval == "step":
                scheduler.step()
        if interval == "epoch":
            scheduler.step()
    torch.testing.assert_close(model.weight, reference)
    if tensor_lr:
        assert trainer.optimizers[0].param_groups[0]["lr"] is model.initial_lr
        snapshot = list(history)
        model.initial_lr.fill_(0)
        assert history == snapshot
