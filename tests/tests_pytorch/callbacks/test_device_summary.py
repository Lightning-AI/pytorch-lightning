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
from unittest import mock

import pytest

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import DeviceSummary
from lightning.pytorch.demos.boring_classes import BoringModel


def test_device_summary_enabled_by_default(tmp_path):
    """Test that DeviceSummary callback is enabled by default."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    device_summary_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, DeviceSummary)]
    assert len(device_summary_callbacks) == 1


def test_device_summary_disabled(tmp_path):
    """Test that DeviceSummary callback can be disabled."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_device_summary=False,
    )
    device_summary_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, DeviceSummary)]
    assert len(device_summary_callbacks) == 0


def test_device_summary_custom_callback(tmp_path):
    """Test that custom DeviceSummary callback is used when provided."""
    custom_callback = DeviceSummary(show_warnings=False)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        callbacks=[custom_callback],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    device_summary_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, DeviceSummary)]
    assert len(device_summary_callbacks) == 1
    assert device_summary_callbacks[0] is custom_callback


def test_device_summary_logs_once(tmp_path):
    """Test that DeviceSummary only logs once per Trainer instance."""
    model = BoringModel()
    callback = DeviceSummary()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[callback],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_device_summary=False,  # Don't add default callback
    )

    with mock.patch.object(callback, "_log_device_info") as mock_log:
        trainer.fit(model)
        assert mock_log.call_count == 1

        # Run validation - should not log again
        trainer.validate(model)
        assert mock_log.call_count == 1


@mock.patch("lightning.pytorch.callbacks.device_summary.rank_zero_info")
def test_device_summary_output(mock_info, tmp_path):
    """Test that DeviceSummary logs expected information."""
    model = BoringModel()
    callback = DeviceSummary()

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=1,
        callbacks=[callback],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_device_summary=False,
    )

    trainer.fit(model)

    # Check that GPU and TPU info was logged
    calls = [str(call) for call in mock_info.call_args_list]
    gpu_logged = any("GPU available" in call for call in calls)
    tpu_logged = any("TPU available" in call for call in calls)
    assert gpu_logged
    assert tpu_logged


def test_device_summary_show_warnings_disabled(tmp_path):
    """Test that warnings can be suppressed."""
    callback = DeviceSummary(show_warnings=False)
    assert callback._show_warnings is False


def test_device_summary_barebones_mode_raises(tmp_path):
    """Test that enable_device_summary raises error in barebones mode."""
    with pytest.raises(ValueError, match="barebones=True, enable_device_summary"):
        Trainer(
            default_root_dir=tmp_path,
            barebones=True,
            enable_device_summary=True,
        )


def test_device_summary_barebones_mode_disabled(tmp_path):
    """Test that DeviceSummary is disabled in barebones mode."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        barebones=True,
    )
    device_summary_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, DeviceSummary)]
    assert len(device_summary_callbacks) == 0
