import math
import os
from datetime import timedelta

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel


class TinyDataset(Dataset):
    def __init__(self, n: int = 4):
        self.x = torch.arange(n, dtype=torch.float32).view(-1, 1)
        self.y = self.x.clone()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TrainMetricModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self._counter = 0.0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = F.mse_loss(y_hat, y)
        # strictly increasing train metric per step
        self._counter += 1.0
        self.log("train_score", torch.tensor(self._counter), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _make_loaders(n=4):
    ds = TinyDataset(n=n)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False)
    return train_loader, val_loader


def test_model_checkpoint_every_n_train_steps_with_train_metric_saves_at_step(tmp_path):
    """When monitoring a train-step metric, step-interval checkpointing should save at the step boundary (no deferral)
    and best_model_score should match the last train metric value."""
    seed_everything(123)

    train_loader, val_loader = _make_loaders(n=4)
    model = TrainMetricModule()

    ckpt = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="train_score",
        mode="max",
        save_top_k=1,
        every_n_train_steps=1,
        train_time_interval=None,
        every_n_epochs=0,
        save_on_train_epoch_end=False,
        save_weights_only=True,
    )

    # 2 batches/epoch, run 2 epochs to have multiple step saves
    trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        callbacks=[ckpt],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        limit_train_batches=2,
        limit_val_batches=0,  # no validation needed for this test
        enable_checkpointing=True,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert ckpt.best_model_score is not None
    # 2 epochs * 2 steps/epoch = 4 steps total; metric increments by 1 each step
    expected = 4.0
    actual = float(ckpt.best_model_score)
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)


@pytest.mark.parametrize("val_scores", [[0.2, 0.4, 0.9]])
def test_model_checkpoint_time_interval_with_val_metric_defers_until_validation(tmp_path, val_scores):
    """With time-interval-based checkpointing, and a validation-only metric, ensure we don't save using stale metrics
    at step boundaries; saving should occur at validation end."""
    seed_everything(123)

    train_loader, val_loader = _make_loaders(n=4)

    model = ValMetricModule(val_scores=val_scores)

    ckpt = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="auroc",
        mode="max",
        save_top_k=1,
        every_n_train_steps=0,  # disable step-based
        train_time_interval=timedelta(seconds=0),  # trigger as often as possible
        every_n_epochs=0,
        save_on_train_epoch_end=False,
        save_weights_only=True,
    )

    trainer = Trainer(
        max_epochs=len(val_scores),
        accelerator="cpu",
        devices=1,
        callbacks=[ckpt],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        limit_train_batches=2,
        limit_val_batches=1,
        enable_checkpointing=True,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert ckpt.best_model_score is not None
    expected = max(val_scores)
    actual = float(ckpt.best_model_score)
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)


class ValMetricModule(LightningModule):
    def __init__(self, val_scores: list[float]):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self._val_scores = [float(s) for s in val_scores]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        score = self._val_scores[self.current_epoch]
        self.log("auroc", torch.tensor(score, dtype=torch.float32), prog_bar=False, logger=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.parametrize("val_scores", [[0.1, 0.5, 1.0, 3.0]])
def test_model_checkpoint_defer_until_next_validation_when_val_every_2_epochs(tmp_path, val_scores):
    """With validation running every 2 epochs, step-triggered saves at the end of non-validation epochs should be
    deferred and then performed at the next validation end when the metric is available."""
    seed_everything(123)

    train_loader, val_loader = _make_loaders(n=4)

    model = ValMetricModule(val_scores=val_scores)

    ckpt = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="auroc",
        mode="max",
        save_top_k=1,
        every_n_train_steps=2,  # end of each epoch
        train_time_interval=None,
        every_n_epochs=0,
        save_on_train_epoch_end=False,
        save_weights_only=True,
    )

    trainer = Trainer(
        max_epochs=len(val_scores),
        accelerator="cpu",
        devices=1,
        callbacks=[ckpt],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        limit_train_batches=2,
        limit_val_batches=1,
        enable_checkpointing=True,
        enable_model_summary=False,
        logger=False,
        check_val_every_n_epoch=2,  # only validate every 2 epochs
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert ckpt.best_model_score is not None
    expected = max(val_scores)  # last/maximum value occurs at final validation epoch
    actual = float(ckpt.best_model_score)
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)


def test_model_checkpoint_save_last_link_symlink_bug(tmp_path):
    """Reproduce the bug where save_last='link' and save_top_k=-1 creates a recursive symlink."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        callbacks=[ModelCheckpoint(dirpath=tmp_path, every_n_epochs=10, save_last="link", save_top_k=-1)],
        enable_checkpointing=True,
        enable_model_summary=False,
        logger=False,
    )

    model = BoringModel()
    trainer.fit(model)

    last_ckpt = tmp_path / "last.ckpt"
    assert last_ckpt.exists()
    # With the fix, if a symlink exists, it should not point to itself (preventing recursion)
    if os.path.islink(str(last_ckpt)):
        assert os.readlink(str(last_ckpt)) != str(last_ckpt)
