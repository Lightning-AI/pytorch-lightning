import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks import GradientStatsMonitor


# -------------------------
# Dummy model
# -------------------------
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


# -------------------------
# Dummy data
# -------------------------
x = torch.randn(32, 10) * 1e1
y = torch.randn(32, 1) * 1e1
loader = DataLoader(TensorDataset(x, y), batch_size=8)

# -------------------------
# Callback
# -------------------------
grad_monitor = GradientStatsMonitor(
    log_every_n_steps=1,
    per_layer=True,
    track_stats=True,
    track_sparsity=True,
    log_histogram=False,  # keep False if you don't want TensorBoard for now
)

# -------------------------
# Lightning Trainer
# -------------------------
trainer = pl.Trainer(
    max_epochs=2,
    limit_train_batches=2,  # just for quick test
    callbacks=[grad_monitor],
    logger=True,  # Lightning will print logs to console
)


# -------------------------
# Override log_dict to print
# -------------------------
class PrintModel(SimpleModel):
    def log_dict(self, metrics, *args, **kwargs):
        print("\nLogged metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        super().log_dict(metrics, *args, **kwargs)


# -------------------------
# Run training
# -------------------------
model = PrintModel()
trainer.fit(model, loader)
