from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import GradientStatsMonitor


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


def test_log_every_n_steps_zero_raises():
    with pytest.raises(ValueError, match="logs nothing"):
        GradientStatsMonitor(log_every_n_steps=0, track_epochs=False)


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

    assert "train/grad_norm" not in all_logged
    assert "train_epoch/grad_norm" in all_logged


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
    assert all_logged["train/grad_norm"] > gradient_clip_val


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


def test_gradient_stats_math():
    cb = GradientStatsMonitor(track_sparsity=False)
    grads = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0])}
    metrics = cb.compute_batch_stats(grads)

    assert metrics["train/grad_norm"] == pytest.approx(5.0)  # sqrt(9 + 16 + 0)
    assert metrics["train/grad_mean"] == pytest.approx(7.0 / 3.0)
    expected_std = (25.0 / 3.0 - (7.0 / 3.0) ** 2) ** 0.5  # sqrt(E[g²] − mean²)
    assert metrics["train/grad_std"] == pytest.approx(expected_std)


def test_compute_epoch_stats_empty():
    cb = GradientStatsMonitor()
    assert cb.compute_epoch_stats(cb.init_epoch_stats()) is None


def test_explosion_warning_not_triggered(tmp_path):
    cb = GradientStatsMonitor(explosion_threshold=1e10)
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        callbacks=[cb],
        default_root_dir=tmp_path,
        enable_checkpointing=False,
        logger=True,
    )
    with patch("lightning.pytorch.callbacks.gradient_statistics_monitor.rank_zero_warn") as mock_warn:
        trainer.fit(SimpleModel(), get_dataloader())
    mock_warn.assert_not_called()
