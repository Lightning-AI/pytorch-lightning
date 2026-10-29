import math
from datetime import timedelta

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint


class TinyDataset(Dataset):
    def __init__(self, n: int = 8):
        self.x = torch.arange(n, dtype=torch.float32).view(-1, 1)
        self.y = self.x.clone()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _make_loaders(n=8, batch_size=2):
    ds = TinyDataset(n=n)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class MultiValPerEpochModule(LightningModule):
    """Logs a validation metric on every validation run, even if validation is run multiple times per epoch."""

    def __init__(self, val_scores: list[float]):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self._val_scores = [float(s) for s in val_scores]
        self._val_call_idx = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        score = self._val_scores[self._val_call_idx]
        self._val_call_idx += 1
        self.log("auroc", torch.tensor(score, dtype=torch.float32), prog_bar=False, logger=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class ValOnceEveryTwoEpochsModule(LightningModule):
    """Logs a validation metric only when validation runs (e.g., every 2 epochs), indexed by current_epoch."""

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
        # current_epoch indexes into provided scores; only called when validation runs
        score = self._val_scores[self.current_epoch]
        self.log("auroc", torch.tensor(score, dtype=torch.float32), prog_bar=False, logger=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.parametrize("val_scores", [[0.1, 0.9]])
def test_checkpoint_defers_with_mid_epoch_validation(tmp_path, val_scores):
    """With val_check_interval=0.5 (validation mid-epoch and at epoch end), and step-based checkpointing, saves must be
    deferred until each validation end so monitored validation metrics are fresh."""
    seed_everything(123)

    # 4 train batches per epoch (batch_size=2 over n=8), so two validations: after 2 batches and after 4 batches
    train_loader, val_loader = _make_loaders(n=8, batch_size=2)

    model = MultiValPerEpochModule(val_scores=val_scores)

    ckpt = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="auroc",
        mode="max",
        save_top_k=1,
        every_n_train_steps=1,  # would trigger every step, but must defer to validation
        train_time_interval=None,
        every_n_epochs=0,
        save_on_train_epoch_end=False,
        save_weights_only=True,
    )

    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        callbacks=[ckpt],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        limit_train_batches=4,  # ensure exactly 4 steps => two validations at 0.5 and 1.0
        limit_val_batches=1,
        enable_checkpointing=True,
        enable_model_summary=False,
        logger=False,
        val_check_interval=0.5,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert ckpt.best_model_score is not None
    expected = max(val_scores)
    actual = float(ckpt.best_model_score)
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)


@pytest.mark.parametrize("val_scores", [[0.2, 0.6]])
def test_time_interval_defers_across_epoch_until_first_validation(tmp_path, val_scores):
    """With time-interval saving and validation only every 2 epochs, ensure no save uses stale/missing validation
    metrics; the first save should happen at the first validation end (epoch 2)."""
    seed_everything(123)

    train_loader, val_loader = _make_loaders(n=4, batch_size=2)

    model = ValOnceEveryTwoEpochsModule(val_scores=val_scores)

    ckpt = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="auroc",
        mode="max",
        save_top_k=1,
        every_n_train_steps=0,  # disable step-based
        train_time_interval=timedelta(seconds=0),  # trigger frequently
        every_n_epochs=0,
        save_on_train_epoch_end=False,
        save_weights_only=True,
    )

    trainer = Trainer(
        max_epochs=2,
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
        check_val_every_n_epoch=2,  # first validation only after 2nd epoch
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert ckpt.best_model_score is not None
    expected = val_scores[1]  # validation runs only once at epoch 2, logging index 1
    actual = float(ckpt.best_model_score)
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6)


@pytest.mark.parametrize(
    ("accumulate_grad_batches", "every_n_train_steps", "crash_at", "resume_step"),
    [
        (2, 2, 5, 2),
        (2, 1, 3, 1),
        (3, 1, 4, 1),
        # without accumulation, every batch is an optimizer step, so the checkpoint can't be saved twice
        (1, 2, 5, 2),
    ],
)
def test_checkpoint_is_not_overwritten_when_resuming_before_optimizer_step(
    tmp_path, accumulate_grad_batches, every_n_train_steps, crash_at, resume_step
):
    """Test that resuming gradient-accumulated training doesn't overwrite the step checkpoint it resumed from with the
    progress of batches whose gradients were not applied yet, which would skip them on the next resume."""

    class Model(LightningModule):
        def __init__(self, crash_at=None):
            super().__init__()
            torch.manual_seed(0)
            self.layer = torch.nn.Linear(4, 1)
            self.crash_at = crash_at

        def training_step(self, batch, batch_idx):
            if batch_idx == self.crash_at:
                raise RuntimeError("simulated crash")
            return (self.layer(batch) - 3).pow(2).mean()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.05)

        def train_dataloader(self):
            return DataLoader(torch.ones(8, 4), batch_size=1)

    def fit(dirpath, crash_at=None, ckpt_path=None):
        model = Model(crash_at)
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            accumulate_grad_batches=accumulate_grad_batches,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[ModelCheckpoint(dirpath=dirpath, every_n_train_steps=every_n_train_steps, save_top_k=-1)],
        )
        if crash_at is None:
            trainer.fit(model, ckpt_path=ckpt_path)
        else:
            with pytest.raises(RuntimeError, match="simulated crash"):
                trainer.fit(model, ckpt_path=ckpt_path)
        return model

    uninterrupted = fit(tmp_path / "uninterrupted")

    checkpoint_path = tmp_path / "run" / f"epoch=0-step={resume_step}.ckpt"
    fit(tmp_path / "run", crash_at=crash_at)
    checkpoint_before_resume = checkpoint_path.read_bytes()

    # crash again before the next optimizer step: the checkpoint we resumed from must not be saved again
    fit(tmp_path / "run", crash_at=crash_at, ckpt_path=checkpoint_path)
    assert checkpoint_path.read_bytes() == checkpoint_before_resume

    # resuming from it once more must not skip any batch
    resumed = fit(tmp_path / "run", ckpt_path=checkpoint_path)
    torch.testing.assert_close(resumed.layer.weight, uninterrupted.layer.weight)
    torch.testing.assert_close(resumed.layer.bias, uninterrupted.layer.bias)


def test_validation_checkpoint_is_saved_when_resuming_before_optimizer_step(tmp_path):
    """Test that the resume guard against saving the restored step again doesn't skip a validation checkpoint at that
    step, which the uninterrupted run would have saved."""

    class Model(LightningModule):
        def __init__(self, crash_at=None):
            super().__init__()
            self.layer = torch.nn.Linear(4, 1)
            self.crash_at = crash_at

        def training_step(self, batch, batch_idx):
            if batch_idx == self.crash_at:
                raise RuntimeError("simulated crash")
            return self.layer(batch).sum()

        def validation_step(self, batch, batch_idx):
            self.log("val_loss", self.layer(batch).sum())

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

        def train_dataloader(self):
            return DataLoader(torch.ones(8, 4), batch_size=1)

        def val_dataloader(self):
            return DataLoader(torch.ones(2, 4), batch_size=1)

    def fit(crash_at=None, ckpt_path=None):
        trainer = Trainer(
            accelerator="cpu",
            max_epochs=1,
            accumulate_grad_batches=2,
            val_check_interval=5,  # validation runs after batch 4, before the optimizer step at batch 5
            num_sanity_val_steps=0,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                ModelCheckpoint(dirpath=tmp_path / "backup", every_n_train_steps=2, save_top_k=-1),
                ModelCheckpoint(
                    dirpath=tmp_path / "val", monitor="val_loss", save_top_k=-1, save_on_train_epoch_end=False
                ),
            ],
        )
        if crash_at is None:
            trainer.fit(Model(), ckpt_path=ckpt_path)
        else:
            with pytest.raises(RuntimeError, match="simulated crash"):
                trainer.fit(Model(crash_at))

    # crash after the step 2 backup is saved, but before validation runs
    fit(crash_at=4)
    assert not (tmp_path / "val").exists()

    fit(ckpt_path=tmp_path / "backup" / "epoch=0-step=2.ckpt")
    assert sorted(p.name for p in (tmp_path / "val").iterdir()) == ["epoch=0-step=2.ckpt"]
