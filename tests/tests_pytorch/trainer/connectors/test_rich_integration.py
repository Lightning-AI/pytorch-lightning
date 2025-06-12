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

from unittest.mock import patch

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelSummary, ProgressBar, RichModelSummary, RichProgressBar, TQDMProgressBar


class TestRichIntegration:
    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", False)
    def test_no_rich_defaults_tqdm_and_model_summary(self, tmp_path):
        trainer = Trainer(default_root_dir=tmp_path, logger=False, enable_checkpointing=False)
        assert any(isinstance(cb, TQDMProgressBar) for cb in trainer.callbacks)
        assert any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)
        assert not any(isinstance(cb, RichProgressBar) for cb in trainer.callbacks)
        assert not any(isinstance(cb, RichModelSummary) for cb in trainer.callbacks)

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", False)
    def test_no_rich_respects_user_provided_tqdm_progress_bar(self, tmp_path):
        user_progress_bar = TQDMProgressBar()
        trainer = Trainer(
            default_root_dir=tmp_path, callbacks=[user_progress_bar], logger=False, enable_checkpointing=False
        )
        assert user_progress_bar in trainer.callbacks
        assert sum(isinstance(cb, ProgressBar) for cb in trainer.callbacks) == 1

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", False)
    def test_no_rich_respects_user_provided_rich_progress_bar(self, tmp_path):
        # If user explicitly provides RichProgressBar, it should be used,
        # even if _RICH_AVAILABLE is False (simulating our connector logic).
        # RequirementCache would normally prevent RichProgressBar instantiation if rich is truly not installed.
        user_progress_bar = RichProgressBar()
        trainer = Trainer(
            default_root_dir=tmp_path, callbacks=[user_progress_bar], logger=False, enable_checkpointing=False
        )
        assert user_progress_bar in trainer.callbacks
        assert sum(isinstance(cb, ProgressBar) for cb in trainer.callbacks) == 1
        assert isinstance(trainer.progress_bar_callback, RichProgressBar)

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", False)
    def test_no_rich_respects_user_provided_model_summary(self, tmp_path):
        user_model_summary = ModelSummary()
        trainer = Trainer(
            default_root_dir=tmp_path, callbacks=[user_model_summary], logger=False, enable_checkpointing=False
        )
        assert user_model_summary in trainer.callbacks
        assert sum(isinstance(cb, ModelSummary) for cb in trainer.callbacks) == 1
        # Check that the specific instance is the one from the trainer's list of ModelSummary callbacks
        assert trainer.callbacks[trainer.callbacks.index(user_model_summary)] == user_model_summary

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", False)
    def test_no_rich_respects_user_provided_rich_model_summary(self, tmp_path):
        user_model_summary = RichModelSummary()
        trainer = Trainer(
            default_root_dir=tmp_path, callbacks=[user_model_summary], logger=False, enable_checkpointing=False
        )
        assert user_model_summary in trainer.callbacks
        assert sum(isinstance(cb, ModelSummary) for cb in trainer.callbacks) == 1
        # Check that the specific instance is the one from the trainer's list of ModelSummary callbacks
        model_summary_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, ModelSummary)]
        assert user_model_summary in model_summary_callbacks
        assert isinstance(model_summary_callbacks[0], RichModelSummary)

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", True)
    def test_rich_available_defaults_rich_progress_and_summary(self, tmp_path):
        trainer = Trainer(default_root_dir=tmp_path, logger=False, enable_checkpointing=False)
        assert any(isinstance(cb, RichProgressBar) for cb in trainer.callbacks)
        assert any(isinstance(cb, RichModelSummary) for cb in trainer.callbacks)
        assert not any(isinstance(cb, TQDMProgressBar) for cb in trainer.callbacks)
        # Ensure the only ModelSummary is the RichModelSummary
        model_summaries = [cb for cb in trainer.callbacks if isinstance(cb, ModelSummary)]
        assert len(model_summaries) == 1
        assert isinstance(model_summaries[0], RichModelSummary)

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", True)
    def test_rich_available_respects_user_tqdm_progress_bar(self, tmp_path):
        user_progress_bar = TQDMProgressBar()
        trainer = Trainer(
            default_root_dir=tmp_path, callbacks=[user_progress_bar], logger=False, enable_checkpointing=False
        )
        assert user_progress_bar in trainer.callbacks
        assert sum(isinstance(cb, ProgressBar) for cb in trainer.callbacks) == 1
        assert isinstance(trainer.progress_bar_callback, TQDMProgressBar)

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", True)
    def test_rich_available_respects_user_model_summary(self, tmp_path):
        user_model_summary = ModelSummary()  # Non-rich
        trainer = Trainer(
            default_root_dir=tmp_path, callbacks=[user_model_summary], logger=False, enable_checkpointing=False
        )
        assert user_model_summary in trainer.callbacks
        model_summaries = [cb for cb in trainer.callbacks if isinstance(cb, ModelSummary)]
        assert len(model_summaries) == 1
        assert isinstance(model_summaries[0], ModelSummary)
        assert not isinstance(model_summaries[0], RichModelSummary)

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", False)
    def test_progress_bar_disabled_no_rich(self, tmp_path):
        trainer = Trainer(
            default_root_dir=tmp_path, enable_progress_bar=False, logger=False, enable_checkpointing=False
        )
        assert not any(isinstance(cb, ProgressBar) for cb in trainer.callbacks)

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", True)
    def test_progress_bar_disabled_with_rich(self, tmp_path):
        trainer = Trainer(
            default_root_dir=tmp_path, enable_progress_bar=False, logger=False, enable_checkpointing=False
        )
        assert not any(isinstance(cb, ProgressBar) for cb in trainer.callbacks)

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", False)
    def test_model_summary_disabled_no_rich(self, tmp_path):
        trainer = Trainer(
            default_root_dir=tmp_path, enable_model_summary=False, logger=False, enable_checkpointing=False
        )
        assert not any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)

    @patch("lightning.pytorch.trainer.connectors.callback_connector._RICH_AVAILABLE", True)
    def test_model_summary_disabled_with_rich(self, tmp_path):
        trainer = Trainer(
            default_root_dir=tmp_path, enable_model_summary=False, logger=False, enable_checkpointing=False
        )
        assert not any(isinstance(cb, ModelSummary) for cb in trainer.callbacks)
