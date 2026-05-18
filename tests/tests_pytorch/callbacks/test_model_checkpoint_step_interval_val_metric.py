import math

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
