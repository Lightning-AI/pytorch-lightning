import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl
from lightning.pytorch import Trainer
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
        loss = torch.nn.functional.mse_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def get_dataloader():
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    return DataLoader(TensorDataset(x, y), batch_size=8)


# -------------------------
# 1. State key test
# -------------------------
def test_gradient_stats_state_key():
    cb = GradientStatsMonitor(per_layer=True)
    assert "GradientStatsMonitor" in cb.state_key
    assert "per_layer" in cb.state_key


# -------------------------
# 2. Runs without crashing
# -------------------------
def test_gradient_stats_runs(tmp_path):
    model = SimpleModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor()

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )

    trainer.fit(model, loader)


# -------------------------
# 3. Logging is triggered
# -------------------------
def test_gradient_logging_called(tmp_path, mocker):
    model = SimpleModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor(log_every_n_steps=1)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )
    
    spy = mocker.spy(model, "log_dict")

    trainer.fit(model, loader)

    assert spy.call_count > 0


# -------------------------
# 4. Per-layer logging works
# -------------------------
def test_per_layer_logging(tmp_path, mocker):
    model = SimpleModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor(per_layer=True)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )

    spy = mocker.spy(model, "log_dict")

    trainer.fit(model, loader)

    logged = spy.call_args[0][0]
    assert any("grad/" in k for k in logged.keys())


# -------------------------
# 5. Stats computation (mean/std exist)
# -------------------------
def test_gradient_stats_values(tmp_path, mocker):
    model = SimpleModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor(track_stats=True)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )

    spy = mocker.spy(model, "log_dict")

    trainer.fit(model, loader)

    logged = spy.call_args[0][0]

    assert "grad/mean" in logged
    assert "grad/std" in logged


# -------------------------
# 6. Sparsity computation
# -------------------------
def test_gradient_sparsity(tmp_path, mocker):
    model = SimpleModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor(track_sparsity=True)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )

    spy = mocker.spy(model, "log_dict")

    trainer.fit(model, loader)

    logged = spy.call_args[0][0]

    assert "grad/sparsity" in logged


# -------------------------
# 7. No gradients edge case
# -------------------------
class NoGradModel(SimpleModel):
    def on_after_backward(self):
        # simulate no gradients
        for p in self.parameters():
            p.grad = None


def test_no_gradients_does_not_crash(tmp_path):
    model = NoGradModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor()

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )

    trainer.fit(model, loader)