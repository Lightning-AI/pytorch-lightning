import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint


class TinyDataset(Dataset):
    def __init__(self, n: int = 4):
        self.x = torch.arange(n, dtype=torch.float32).view(-1, 1)
        self.y = self.x.clone()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ValMetricModule(LightningModule):
    def __init__(self, val_scores: list[float]):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self._val_scores = [float(s) for s in val_scores]

    # LightningModule API (minimal)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # do nothing per-step; we log at epoch end
        pass

    def on_validation_epoch_end(self):
        # Log a validation metric only at validation epoch end
        # Values increase across epochs; best should be the last epoch
        score = self._val_scores[self.current_epoch]
        # use logger=True so it lands in trainer.callback_metrics
        self.log("auroc", torch.tensor(score, dtype=torch.float32), prog_bar=False, logger=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.parametrize("val_scores", [[0.1, 0.5, 1.0]])
def test_model_checkpoint_every_n_train_steps_with_val_metric_saves_after_val(tmp_path, val_scores):
    """Reproduces #20919: Using every_n_train_steps with a validation-only metric should save the best checkpoint only
    after the metric is computed at validation, not earlier at the train-step boundary.

    Expectation: best_model_score equals the last (max) val score.

    """
    seed_everything(123)

    # 2 train batches per epoch (so checkpoint triggers at the epoch boundary)
    ds = TinyDataset(n=4)
    train_loader = DataLoader(ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(ds, batch_size=2, shuffle=False)

    model = ValMetricModule(val_scores=val_scores)

    ckpt = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="auroc",
        mode="max",
        save_top_k=1,
        # critical: trigger on train steps, not on epoch end
        every_n_train_steps=2,  # equal to number of train batches per epoch
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
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert ckpt.best_model_score is not None
    # Should equal the last (max) validation score
    expected = max(val_scores)
    actual = float(ckpt.best_model_score)
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6), (
        f"best_model_score should be {expected} (last/maximum val score), got {actual}.\n"
        f"This indicates the checkpoint was saved before the validation metric was computed."
    )


class StepScoreModule(LightningModule):
    """Logs a validation-only metric equal to the global step at which validation runs."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.mse_loss(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        self.log("score", float(self.global_step))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.parametrize(
    ("n", "trainer_kwargs"),
    [
        # validation once per epoch (the defaults), so `save_on_train_epoch_end` resolves to True
        (4, {"max_epochs": 3}),
        # iteration-based validation, mid-epoch (#20185)
        (16, {"max_steps": 8, "check_val_every_n_epoch": None, "val_check_interval": 2}),
        # validation twice per epoch
        (8, {"max_epochs": 2, "val_check_interval": 0.5}),
    ],
)
def test_step_checkpoint_scored_with_validation_that_follows_it(tmp_path, n, trainer_kwargs):
    """Test that a step checkpoint saved right before validation is ranked with the metric of that validation, not with
    the one of the previous validation."""
    loader = DataLoader(TinyDataset(n=n), batch_size=2)
    ckpt = ModelCheckpoint(
        dirpath=tmp_path,
        filename="{step}",
        auto_insert_metric_name=False,
        monitor="score",
        mode="max",
        save_top_k=-1,
        every_n_train_steps=2,
    )
    trainer = Trainer(
        callbacks=[ckpt],
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **trainer_kwargs,
    )
    trainer.fit(StepScoreModule(), train_dataloaders=loader, val_dataloaders=loader)

    scores = {int(Path(path).stem): float(score) for path, score in ckpt.best_k_models.items()}
    assert scores == {step: float(step) for step in range(2, trainer.global_step + 1, 2)}
    assert ckpt.best_model_path == str(tmp_path / f"{trainer.global_step}.ckpt")


@pytest.mark.parametrize("save_on_train_epoch_end", [None, True])
def test_step_checkpoint_with_train_epoch_metric(tmp_path, save_on_train_epoch_end):
    """Test that monitoring a train epoch metric with `every_n_train_steps` doesn't fail when validation runs before
    the metric is logged at the end of the first epoch."""

    class TrainEpochMetricModule(StepScoreModule):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            self.log("train_loss", loss, on_step=False, on_epoch=True)
            return loss

    loader = DataLoader(TinyDataset(n=4), batch_size=2)
    ckpt = ModelCheckpoint(
        dirpath=tmp_path,
        filename="{step}",
        auto_insert_metric_name=False,
        monitor="train_loss",
        save_top_k=-1,
        every_n_train_steps=2,
        save_on_train_epoch_end=save_on_train_epoch_end,
    )
    trainer = Trainer(
        max_epochs=3,
        callbacks=[ckpt],
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(TrainEpochMetricModule(), train_dataloaders=loader, val_dataloaders=loader)

    assert str(tmp_path / f"{trainer.global_step}.ckpt") in ckpt.best_k_models
