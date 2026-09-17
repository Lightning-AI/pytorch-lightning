# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for TQDMProgressBar.warmup_batches — issue #19357.

These tests verify that the progress bar timer is correctly reset after warmup batches so that torch.compile (and other
one-time startup costs) do not skew the displayed it/s rate and ETA.

"""

import time
from unittest.mock import patch

import pytest

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from lightning.pytorch.demos.boring_classes import BoringModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SlowFirstBatchModel(BoringModel):
    """BoringModel whose first batch takes noticeably longer than later ones.

    A small ``sleep_seconds`` value is sufficient because we only need to
    confirm that ``start_t`` moves forward — we do not need real-world
    compile latency in unit tests.

    """

    def __init__(self, sleep_seconds: float = 0.05) -> None:
        super().__init__()
        self._sleep_seconds = sleep_seconds
        self._batch_call_count = 0

    def training_step(self, batch, batch_idx):
        if self._batch_call_count == 0:
            time.sleep(self._sleep_seconds)
        self._batch_call_count += 1
        return super().training_step(batch, batch_idx)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_warmup_batches_negative_raises():
    """warmup_batches must be a non-negative integer."""
    with pytest.raises(ValueError, match="non-negative"):
        TQDMProgressBar(warmup_batches=-1)


def test_warmup_batches_zero_is_valid():
    """warmup_batches=0 (the default) must not raise."""
    bar = TQDMProgressBar(warmup_batches=0)
    assert bar._warmup_batches == 0


def test_warmup_batches_positive_stored():
    """warmup_batches value should be stored on the instance."""
    bar = TQDMProgressBar(warmup_batches=3)
    assert bar._warmup_batches == 3


# ---------------------------------------------------------------------------
# Timer reset behaviour
# ---------------------------------------------------------------------------


def test_tqdm_reset_timer_moves_start_t():
    """Tqdm.reset_timer() must advance start_t to approximately now."""
    bar = Tqdm(total=100, disable=True)
    original_start_t = bar.start_t

    # Sleep a tiny amount so that the new timestamp is strictly greater.
    time.sleep(0.01)
    bar.reset_timer()

    assert bar.start_t > original_start_t, "reset_timer() should update start_t to a later time"
    bar.close()


def test_tqdm_reset_timer_zeroes_ema_fields():
    """reset_timer() should zero out any EMA accumulator fields that exist."""
    bar = Tqdm(total=100, disable=True)

    # Simulate some iterations so the EMA fields get populated.
    bar.update(10)

    bar.reset_timer()

    for attr in ("_ema_dn", "_ema_dt", "avg_time"):
        if hasattr(bar, attr):
            assert getattr(bar, attr) == 0.0, f"reset_timer() should zero {attr!r}, got {getattr(bar, attr)}"
    bar.close()


def test_tqdm_reset_timer_does_not_change_n():
    """reset_timer() must not alter the iteration counter."""
    bar = Tqdm(total=100, disable=True)
    bar.update(42)
    n_before = bar.n

    bar.reset_timer()

    assert bar.n == n_before, "reset_timer() must not change bar.n"
    bar.close()


# ---------------------------------------------------------------------------
# Integration: warmup_batches=0 (default) — no timer reset
# ---------------------------------------------------------------------------


def test_warmup_batches_zero_no_reset(tmp_path):
    """With warmup_batches=0 reset_timer() should never be called."""
    pbar = TQDMProgressBar(warmup_batches=0)
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[pbar],
        max_epochs=1,
        limit_train_batches=3,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )

    with patch.object(Tqdm, "reset_timer") as mock_reset:
        trainer.fit(model)

    mock_reset.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: warmup_batches=1 — timer resets exactly once on epoch 0
# ---------------------------------------------------------------------------


def test_warmup_batches_resets_once_on_epoch_0(tmp_path):
    """With warmup_batches=1 the timer must be reset exactly once, after batch 0 of epoch 0, regardless of how many
    epochs are trained."""
    pbar = TQDMProgressBar(warmup_batches=1)
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[pbar],
        max_epochs=3,
        limit_train_batches=4,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )

    with patch.object(Tqdm, "reset_timer") as mock_reset:
        trainer.fit(model)

    mock_reset.assert_called_once()


def test_warmup_batches_2_resets_after_second_batch(tmp_path):
    """With warmup_batches=2 the timer must reset after batch index 1 (the 2nd batch)."""
    reset_at_batch_idx = []

    class TrackingProgressBar(TQDMProgressBar):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # Record the batch_idx at the moment reset_timer would fire.
            before = self.train_progress_bar.start_t if isinstance(self.train_progress_bar, Tqdm) else None
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            after = self.train_progress_bar.start_t if isinstance(self.train_progress_bar, Tqdm) else None
            if before is not None and after is not None and after != before:
                reset_at_batch_idx.append(batch_idx)

    pbar = TrackingProgressBar(warmup_batches=2)
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[pbar],
        max_epochs=2,
        limit_train_batches=5,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model)

    # The reset should have happened exactly once, at batch_idx == 1.
    assert reset_at_batch_idx == [1], f"Expected reset at batch_idx=1 only, got resets at: {reset_at_batch_idx}"


# ---------------------------------------------------------------------------
# Integration: warmup_batches > total batches — reset never fires, no error
# ---------------------------------------------------------------------------


def test_warmup_batches_exceeds_total_batches_no_error(tmp_path):
    """If warmup_batches exceeds the number of batches per epoch the bar should function normally without any error or
    unexpected reset."""
    pbar = TQDMProgressBar(warmup_batches=100)
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[pbar],
        max_epochs=1,
        limit_train_batches=3,  # fewer batches than warmup_batches
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )

    with patch.object(Tqdm, "reset_timer") as mock_reset:
        trainer.fit(model)  # must not raise

    mock_reset.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: reset does not affect the progress counter
# ---------------------------------------------------------------------------


def test_warmup_reset_does_not_affect_progress_counter(tmp_path):
    """After the warmup reset the bar's n counter must continue from where it left off, not restart from zero."""
    pbar = TQDMProgressBar(warmup_batches=1)
    model = BoringModel()
    limit_batches = 5
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[pbar],
        max_epochs=1,
        limit_train_batches=limit_batches,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model)

    # After training completes, n should equal the total batches processed.
    assert pbar.train_progress_bar.n == limit_batches, (
        f"Expected bar.n == {limit_batches} after training, got {pbar.train_progress_bar.n}"
    )


# ---------------------------------------------------------------------------
# Integration: timer is genuinely reset — start_t moves forward
# ---------------------------------------------------------------------------


def test_warmup_reset_advances_start_t(tmp_path):
    """The start_t recorded before batch 0 and the one after warmup reset should differ — confirming the timer actually
    moved forward."""
    start_t_values: list = []

    class CapturingProgressBar(TQDMProgressBar):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if batch_idx == 0 and trainer.current_epoch == 0 and isinstance(self.train_progress_bar, Tqdm):
                # Capture start_t before the reset fires (super() call below).
                start_t_values.append(("before", self.train_progress_bar.start_t))
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            if batch_idx == 0 and trainer.current_epoch == 0 and isinstance(self.train_progress_bar, Tqdm):
                start_t_values.append(("after", self.train_progress_bar.start_t))

    pbar = CapturingProgressBar(warmup_batches=1)
    model = SlowFirstBatchModel(sleep_seconds=0.02)
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[pbar],
        max_epochs=1,
        limit_train_batches=3,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model)

    assert len(start_t_values) == 2
    label_before, t_before = start_t_values[0]
    label_after, t_after = start_t_values[1]
    assert label_before == "before"
    assert label_after == "after"
    assert t_after > t_before, (
        f"start_t should be greater after reset_timer() than before, got before={t_before:.6f} after={t_after:.6f}"
    )


# ---------------------------------------------------------------------------
# Serialisation: warmup_batches survives pickling
# ---------------------------------------------------------------------------


def test_warmup_batches_picklable(tmp_path):
    """TQDMProgressBar with warmup_batches set must be picklable (required for multi-process / DDP usage)."""
    import pickle

    pbar = TQDMProgressBar(warmup_batches=2)
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        callbacks=[pbar],
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model)

    # Should not raise.
    data = pickle.dumps(pbar)
    restored = pickle.loads(data)
    assert restored._warmup_batches == 2
