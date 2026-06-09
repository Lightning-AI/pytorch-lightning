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
"""Test for manual dataloader reloading feature (issue #21448)."""

import pytest
import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Callback, LightningDataModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset


class ReloadTrackingDataModule(LightningDataModule):
    """DataModule that tracks when train_dataloader is called."""

    def __init__(self, sequence_length: int = 32):
        super().__init__()
        self.sequence_length = sequence_length
        self.train_dataloader_call_count = 0
        self.val_dataloader_call_count = 0
        self._train_epochs_called_for = []
        self._val_epochs_called_for = []

    def train_dataloader(self):
        self.train_dataloader_call_count += 1
        if self.trainer is not None:
            self._train_epochs_called_for.append(self.trainer.current_epoch)
        return DataLoader(RandomDataset(self.sequence_length, 64), batch_size=8)

    def val_dataloader(self):
        self.val_dataloader_call_count += 1
        if self.trainer is not None:
            self._val_epochs_called_for.append(self.trainer.current_epoch)
        return DataLoader(RandomDataset(self.sequence_length, 64), batch_size=8)


class ManualReloadCallback(Callback):
    """Callback that triggers manual dataloader reload at specific epochs."""

    def __init__(self, reload_at_epoch: int, reload_train: bool = True, reload_val: bool = False):
        super().__init__()
        self.reload_at_epoch = reload_at_epoch
        self.reload_train = reload_train
        self.reload_val = reload_val

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.reload_at_epoch:
            trainer.reload_dataloaders(train=self.reload_train, val=self.reload_val)


class MetricBasedReloadCallback(Callback):
    """Callback that triggers reload based on training metrics (curriculum learning example)."""

    def __init__(self, loss_threshold: float = 0.5):
        super().__init__()
        self.loss_threshold = loss_threshold
        self.reload_triggered = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.reload_triggered:
            loss = trainer.callback_metrics.get("train_loss", 1.0)
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            if loss < self.loss_threshold:
                # Update datamodule parameters before reload
                if trainer.datamodule is not None:
                    trainer.datamodule.sequence_length += 10
                trainer.reload_dataloaders()
                self.reload_triggered = True


def test_reload_dataloaders_outside_training_raises_error():
    """Test that calling reload_dataloaders outside of fit() raises RuntimeError."""
    trainer = Trainer(max_epochs=1)

    with pytest.raises(RuntimeError, match="can only be called during training"):
        trainer.reload_dataloaders()


def test_manual_reload_train_dataloader(tmp_path):
    """Test that manually triggering train dataloader reload works."""

    class TrackingModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            return super().validation_step(batch, batch_idx)

    model = TrackingModel()
    dm = ReloadTrackingDataModule()
    callback = ManualReloadCallback(reload_at_epoch=1, reload_train=True, reload_val=False)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=4,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # Train dataloader should be called at epochs 0, 2 (after manual reload at epoch 1)
    # Without the manual reload, it would only be called at epoch 0
    assert dm.train_dataloader_call_count >= 2, (
        f"Expected at least 2 train_dataloader calls, got {dm.train_dataloader_call_count}"
    )


def test_manual_reload_val_dataloader(tmp_path):
    """Test that manually triggering validation dataloader reload works."""

    class TrackingModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            return super().validation_step(batch, batch_idx)

    model = TrackingModel()
    dm = ReloadTrackingDataModule()
    callback = ManualReloadCallback(reload_at_epoch=1, reload_train=False, reload_val=True)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=4,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # Validation dataloader should be called multiple times due to manual reload
    assert dm.val_dataloader_call_count >= 2, (
        f"Expected at least 2 val_dataloader calls, got {dm.val_dataloader_call_count}"
    )


def test_manual_reload_both_dataloaders(tmp_path):
    """Test that manually triggering both train and val dataloader reload works."""

    class TrackingModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            return super().validation_step(batch, batch_idx)

    model = TrackingModel()
    dm = ReloadTrackingDataModule()
    callback = ManualReloadCallback(reload_at_epoch=1, reload_train=True, reload_val=True)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=4,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # Both dataloaders should be called multiple times
    assert dm.train_dataloader_call_count >= 2
    assert dm.val_dataloader_call_count >= 2


def test_manual_reload_updates_datamodule_params(tmp_path):
    """Test that datamodule parameters can be updated before manual reload."""

    class TrackingModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            return super().validation_step(batch, batch_idx)

    class ParamUpdateCallback(Callback):
        def __init__(self):
            super().__init__()
            self.sequence_lengths_seen = []

        def on_train_epoch_start(self, trainer, pl_module):
            if trainer.datamodule is not None:
                self.sequence_lengths_seen.append(trainer.datamodule.sequence_length)

        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch == 1:
                # Update datamodule parameters
                trainer.datamodule.sequence_length = 64
                # Trigger reload
                trainer.reload_dataloaders(train=True)

    model = TrackingModel()
    dm = ReloadTrackingDataModule(sequence_length=32)
    callback = ParamUpdateCallback()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=4,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # After epoch 1, sequence_length should have been updated to 64
    assert dm.sequence_length == 64
    # And it should have been seen in subsequent epochs
    assert 64 in callback.sequence_lengths_seen or dm.train_dataloader_call_count >= 2


def test_reload_dataloaders_from_lightning_module(tmp_path):
    """Test that reload_dataloaders can be called from within the LightningModule."""

    class ReloadingModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.reload_triggered = False

        def on_train_epoch_end(self):
            if self.current_epoch == 1 and not self.reload_triggered:
                self.trainer.reload_dataloaders(train=True)
                self.reload_triggered = True

        def validation_step(self, batch, batch_idx):
            return super().validation_step(batch, batch_idx)

    model = ReloadingModel()
    dm = ReloadTrackingDataModule()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=4,
        limit_train_batches=2,
        limit_val_batches=2,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # Should have triggered reload
    assert dm.train_dataloader_call_count >= 2


def test_reload_dataloaders_multiple_times(tmp_path):
    """Test that reload_dataloaders can be called multiple times."""

    class TrackingModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            return super().validation_step(batch, batch_idx)

    class MultiReloadCallback(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            # Reload at every epoch
            trainer.reload_dataloaders(train=True)

    model = TrackingModel()
    dm = ReloadTrackingDataModule()
    callback = MultiReloadCallback()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=4,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # Train dataloader should be called at every epoch
    # Initial load + 3 reloads (after epochs 0, 1, 2)
    # Note: reload at epoch 3 end won't take effect since training ends
    assert dm.train_dataloader_call_count >= 4


def test_reload_dataloaders_with_reload_every_n_epochs(tmp_path):
    """Test that manual reload works alongside reload_dataloaders_every_n_epochs."""

    class TrackingModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            return super().validation_step(batch, batch_idx)

    model = TrackingModel()
    dm = ReloadTrackingDataModule()
    # Manual reload at epoch 0 (will reload at epoch 1 start)
    callback = ManualReloadCallback(reload_at_epoch=0, reload_train=True)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=4,
        limit_train_batches=2,
        limit_val_batches=2,
        reload_dataloaders_every_n_epochs=3,  # Would reload at epoch 3
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # Should have initial load + manual reload + possibly automatic reload
    assert dm.train_dataloader_call_count >= 2


def test_reload_dataloaders_default_args(tmp_path):
    """Test reload_dataloaders with default arguments (train=True, val=False)."""

    class TrackingModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            return super().validation_step(batch, batch_idx)

    class DefaultArgsCallback(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch == 1:
                # Call with default args
                trainer.reload_dataloaders()

    model = TrackingModel()
    dm = ReloadTrackingDataModule()
    callback = DefaultArgsCallback()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=4,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[callback],
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)

    # Train dataloader should be reloaded
    assert dm.train_dataloader_call_count >= 2
