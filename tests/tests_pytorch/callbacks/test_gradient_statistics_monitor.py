from unittest.mock import MagicMock, patch

import pytest
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
        return torch.nn.functional.mse_loss(self(x), y)

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
def test_gradient_logging_called(tmp_path):
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

    with patch.object(model, "log_dict", wraps=model.log_dict) as spy:
        trainer.fit(model, loader)

    assert spy.call_count > 0


# -------------------------
# 4. Per-layer logging works
# -------------------------
def test_per_layer_logging(tmp_path):
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

    with patch.object(model, "log_dict", wraps=model.log_dict) as spy:
        trainer.fit(model, loader)

    all_logged = {}
    for c in spy.call_args_list:
        all_logged.update(c[0][0])

    assert any("grad/" in k for k in all_logged)


# -------------------------
# 5. Stats computation (mean/std exist)
# -------------------------
def test_gradient_stats_values(tmp_path):
    model = SimpleModel()
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

    with patch.object(model, "log_dict", wraps=model.log_dict) as spy:
        trainer.fit(model, loader)

    all_logged = {}
    for c in spy.call_args_list:
        all_logged.update(c[0][0])

    assert "train/grad/mean" in all_logged
    assert "train/grad/std" in all_logged


# -------------------------
# 6. Sparsity computation
# -------------------------
def test_gradient_sparsity(tmp_path):
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

    with patch.object(model, "log_dict", wraps=model.log_dict) as spy:
        trainer.fit(model, loader)

    all_logged = {}
    for c in spy.call_args_list:
        all_logged.update(c[0][0])

    assert "train/grad/sparsity" in all_logged


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


# -------------------------
# 8. Explosion warning fires when norm exceeds threshold
# -------------------------
def test_explosion_warning_triggered(tmp_path):
    model = SimpleModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor(explosion_threshold=1e-10)  # near-zero threshold — any gradient triggers it

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )

    with patch("lightning.pytorch.callbacks.gradients_statistics_monitor.rank_zero_warn") as mock_warn:
        trainer.fit(model, loader)

    assert mock_warn.called


# -------------------------
# 9. Epoch metrics are logged with correct keys
# -------------------------
def test_epoch_metrics_logged(tmp_path):
    model = SimpleModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor(track_epochs=True, log_every_n_steps=1)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )

    with patch.object(model, "log_dict", wraps=model.log_dict) as spy:
        trainer.fit(model, loader)

    all_logged = {}
    for c in spy.call_args_list:
        all_logged.update(c[0][0])

    assert "train/epoch/grad/global_norm" in all_logged
    assert "train/epoch/grad/mean" in all_logged
    assert "train/epoch/grad/std" in all_logged


# -------------------------
# 10. log_every_n_steps=0 raises when epoch tracking is also disabled
# -------------------------
def test_log_every_n_steps_zero_raises():
    with pytest.raises(ValueError, match="logs nothing"):
        GradientStatsMonitor(log_every_n_steps=0, track_epochs=False)


# -------------------------
# 11. log_every_n_steps=0 suppresses per-batch logging but keeps epoch logging
# -------------------------
def test_log_every_n_steps_zero_disables_batch_logging(tmp_path):
    model = SimpleModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor(log_every_n_steps=0, track_epochs=True)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )

    with patch.object(model, "log_dict", wraps=model.log_dict) as spy:
        trainer.fit(model, loader)

    all_logged = {}
    for c in spy.call_args_list:
        all_logged.update(c[0][0])

    assert "train/grad/global_norm" not in all_logged
    assert "train/epoch/grad/global_norm" in all_logged


# -------------------------
# 12. Non-global-zero rank does not call log_dict
# -------------------------
def test_non_global_zero_does_not_log():
    cb = GradientStatsMonitor()
    model = SimpleModel()

    trainer_mock = MagicMock()
    trainer_mock.is_global_zero = False
    trainer_mock.logger = MagicMock()

    with patch.object(model, "log_dict") as mock_log:
        cb._log_scalars(trainer_mock, model, {"train/grad/global_norm": 1.0})

    mock_log.assert_not_called()


# -------------------------
# 13. Gradients are captured before clipping
# -------------------------
def test_captures_pre_clip_gradients(tmp_path):
    """Logged norms must reflect pre-clip gradients, not post-clip ones."""
    model = SimpleModel()
    loader = get_dataloader()
    gradient_clip_val = 1e-6  # near-zero clip; any natural gradient will exceed it

    cb = GradientStatsMonitor(log_every_n_steps=1, track_epochs=False)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
        gradient_clip_val=gradient_clip_val,
    )

    with patch.object(model, "log_dict", wraps=model.log_dict) as spy:
        trainer.fit(model, loader)

    all_logged = {}
    for c in spy.call_args_list:
        all_logged.update(c[0][0])

    # Pre-clip norm must exceed the clip value; post-clip norm never could.
    assert all_logged["train/grad/global_norm"] > gradient_clip_val


# -------------------------
# 14. state_dict / load_state_dict round-trip
# -------------------------
def test_state_dict_round_trip(tmp_path):
    """load_state_dict must restore epoch accumulator and last-logged-step."""
    model = SimpleModel()
    loader = get_dataloader()

    cb = GradientStatsMonitor(log_every_n_steps=1, track_epochs=True)

    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )
    trainer.fit(model, loader)

    sd = cb.state_dict()
    assert sd["train_stats"]["steps"] > 0

    cb2 = GradientStatsMonitor(log_every_n_steps=1, track_epochs=True)
    cb2.load_state_dict(sd)

    assert cb2._train_stats["steps"] == sd["train_stats"]["steps"]
    assert cb2._last_logged_step == sd["last_logged_step"]
