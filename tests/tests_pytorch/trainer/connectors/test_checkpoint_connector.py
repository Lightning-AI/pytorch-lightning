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
import errno
import operator
import os
import re
from unittest import mock
from unittest.mock import ANY, Mock

import pytest
import torch
from lightning_utilities.core.imports import compare_version

from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.migration.utils import _set_version


def test_preloaded_checkpoint_lifecycle(tmp_path):
    """Tests that the preloaded checkpoint contents gets cleared from memory when it is not required anymore."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, max_steps=1)
    trainer.fit(model)

    connector = trainer._checkpoint_connector

    assert not connector._ckpt_path
    assert not connector._loaded_checkpoint

    connector.resume_start()
    assert not connector._ckpt_path
    assert not connector._loaded_checkpoint
    connector.resume_end()
    assert not connector._ckpt_path
    assert not connector._loaded_checkpoint

    ckpt_path = trainer.checkpoint_callback.best_model_path
    trainer = Trainer(default_root_dir=tmp_path, max_steps=2)
    connector = trainer._checkpoint_connector
    connector.resume_start(ckpt_path)
    assert connector._ckpt_path == ckpt_path
    assert connector._loaded_checkpoint
    assert isinstance(connector._loaded_checkpoint, dict)
    trainer.state.fn = TrainerFn.FITTING
    connector.resume_end()
    # not cleared until next restoration, as the user might access it through `trainer.ckpt_path`
    assert connector._ckpt_path == ckpt_path
    assert not connector._loaded_checkpoint


@mock.patch("lightning.fabric.plugins.environments.slurm.SLURMEnvironment.detect", return_value=True)
def test_hpc_restore_attempt(_, tmp_path):
    """Test that restore() attempts to restore the hpc_ckpt with highest priority."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, max_steps=1, enable_checkpointing=False, logger=False)
    trainer.fit(model)

    hpc_ckpt_path = tmp_path / "hpc_ckpt_3.ckpt"
    trainer.save_checkpoint(hpc_ckpt_path)
    assert os.listdir(tmp_path) == ["hpc_ckpt_3.ckpt"]

    # set weights to zero
    for param in model.parameters():
        torch.nn.init.constant_(param, 0)

    # case 1: restore hpc first, no explicit resume path provided
    trainer = Trainer(default_root_dir=tmp_path, max_steps=2, enable_checkpointing=False, logger=False)
    trainer.fit(model)

    for param in model.parameters():
        assert param.abs().sum() > 0
        torch.nn.init.constant_(param, 0)

    # case 2: explicit resume path provided, file not found
    trainer = Trainer(default_root_dir=tmp_path, max_steps=3)

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found: not existing"):
        trainer.fit(model, ckpt_path="not existing")


def test_hpc_max_ckpt_version(tmp_path):
    """Test that the _CheckpointConnector is able to find the hpc checkpoint file with the highest version."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, max_steps=1)
    trainer.fit(model)
    trainer.save_checkpoint(tmp_path / "hpc_ckpt.ckpt")
    trainer.save_checkpoint(tmp_path / "hpc_ckpt_0.ckpt")
    trainer.save_checkpoint(tmp_path / "hpc_ckpt_3.ckpt")
    trainer.save_checkpoint(tmp_path / "hpc_ckpt_33.ckpt")

    assert trainer._checkpoint_connector._hpc_resume_path == str(tmp_path / "hpc_ckpt_33.ckpt")
    assert trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmp_path) == 33
    assert (
        trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder(tmp_path / "not" / "existing")
        is None
    )


def test_local_cross_device_checkpoint(tmpdir):
    """Test that the _CheckpointConnector can write local cross-device files or raises an error if fsspec<2025.5.0."""
    model = BoringModel()
    # hardcoding dir since `tmp_path` can be windows path
    trainer = Trainer(
        default_root_dir="memory://test_ckpt_for_fsspec", limit_train_batches=1, limit_val_batches=1, max_epochs=1
    )
    trainer.fit(model)
    # Simulate the behavior of fsspec when writing to a local file system but other device.
    with (
        mock.patch("os.rename", side_effect=OSError(errno.EXDEV, "Invalid cross-device link")),
        mock.patch("os.chmod", side_effect=PermissionError("Operation not permitted")),
    ):
        if compare_version("fsspec", operator.lt, "2025.5.0"):
            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    'Upgrade fsspec to enable cross-device local checkpoints: pip install "fsspec[http]>=2025.5.0"'
                ),
            ):
                trainer.save_checkpoint(tmpdir + "/test_ckpt_for_fsspec/hpc_ckpt.ckpt")
        else:
            trainer.save_checkpoint(tmpdir + "/test_ckpt_for_fsspec/hpc_ckpt.ckpt")


def test_ckpt_for_fsspec():
    """Test that the _CheckpointConnector is able to write to fsspec file systems."""
    model = BoringModel()
    # hardcoding dir since `tmp_path` can be windows path
    trainer = Trainer(
        default_root_dir="memory://test_ckpt_for_fsspec", limit_train_batches=1, limit_val_batches=1, max_epochs=1
    )
    trainer.fit(model)
    trainer.save_checkpoint("memory://test_ckpt_for_fsspec/hpc_ckpt.ckpt")
    trainer.save_checkpoint("memory://test_ckpt_for_fsspec/hpc_ckpt_0.ckpt")
    trainer.save_checkpoint("memory://test_ckpt_for_fsspec/hpc_ckpt_3.ckpt")
    trainer.save_checkpoint("memory://test_ckpt_for_fsspec/hpc_ckpt_33.ckpt")

    assert trainer._checkpoint_connector._hpc_resume_path == "memory://test_ckpt_for_fsspec/hpc_ckpt_33.ckpt"
    assert (
        trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder("memory://test_ckpt_for_fsspec")
        == 33
    )
    assert (
        trainer._checkpoint_connector._CheckpointConnector__max_ckpt_version_in_folder("memory://not_existing") is None
    )


def test_loops_restore(tmp_path):
    """Test that required loop state_dict is loaded correctly by checkpoint connector."""
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    trainer_args = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "logger": False,
        "callbacks": [checkpoint_callback],
        "num_sanity_val_steps": 0,
    }
    trainer = Trainer(**trainer_args)
    trainer.fit(model)

    ckpt_path = str(tmp_path / "last.ckpt")

    trainer = Trainer(**trainer_args)
    trainer.strategy.connect(model)

    trainer_fns = list(TrainerFn)
    for fn in trainer_fns:
        trainer_fn = getattr(trainer, f"{fn.value}_loop")
        trainer_fn.load_state_dict = mock.Mock()

    for fn in trainer_fns:
        trainer.state.fn = fn
        trainer._checkpoint_connector.resume_start(ckpt_path)
        trainer._checkpoint_connector.restore_loops()

        trainer_loop = getattr(trainer, f"{fn.value}_loop")
        trainer_loop.load_state_dict.assert_called()
        trainer_loop.load_state_dict.reset_mock()

        for fn2 in trainer_fns:
            if fn2 != fn:
                trainer_loop2 = getattr(trainer, f"{fn2.value}_loop")
                trainer_loop2.load_state_dict.assert_not_called()


def test_stateful_trainer_ckpt_path_support(tmp_path):
    """Tests support for the pattern used by NeMo's experiment manager."""
    model = BoringModel()

    # dummy ckpt data
    ckpt_data = {"state_dict": model.state_dict(), "optimizer_states": {}, "lr_schedulers": {}}
    _set_version(ckpt_data, "2.0.0")

    # save a "checkpoint"
    ckpt_path = tmp_path / "foo.ckpt"
    torch.save(ckpt_data, ckpt_path)

    # mock model checkpoint instance that has saved a last checkpoint
    model_checkpoint = Mock(spec=ModelCheckpoint)
    last_path = tmp_path / "last.ckpt"
    torch.save(ckpt_data, last_path)
    model_checkpoint._find_last_checkpoints.return_value = {last_path}

    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, callbacks=model_checkpoint)

    # set the ckpt path statefully
    trainer.ckpt_path = ckpt_path
    trainer.fit(model)
    assert trainer.ckpt_path == ckpt_path  # not automatically cleaned
    assert trainer._checkpoint_connector._user_managed

    # now conflict with ckpt_path functionally
    with pytest.warns(UserWarning, match="trainer.ckpt_path =.*but then you passed"):
        trainer.fit(model, ckpt_path="last")
    assert trainer.ckpt_path == last_path
    assert not trainer._checkpoint_connector._user_managed

    # mock model checkpoint instance that has saved a last checkpoint
    best_path = tmp_path / "best.ckpt"
    torch.save(ckpt_data, best_path)
    model_checkpoint.best_model_path = best_path

    # `trainer.test` will use this over "best" if statefully set
    trainer.ckpt_path = ckpt_path
    trainer.test()
    assert trainer.ckpt_path == ckpt_path

    # ckpt_path = "best" still works if it's reset
    trainer.ckpt_path = None
    # the state is cleared
    assert trainer._checkpoint_connector._ckpt_path is None
    assert not trainer._checkpoint_connector._user_managed
    trainer.test()
    assert trainer.ckpt_path == best_path


@pytest.mark.parametrize("trainer_fn", ["validate", "test", "predict"])
def test_best_ckpt_path_from_disk_in_fresh_process(tmp_path, trainer_fn):
    """Test that ckpt_path="best" is resolved from disk when training happened in a separate process.

    Regression test for https://github.com/Lightning-AI/pytorch-lightning/issues/21254. A fresh Trainer/process has an
    empty in-memory ``best_model_path``, so ``"best"`` must be recovered from the persisted callback state on disk.

    """

    class LogValLossModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("val_loss", loss)
            return {"x": loss}

    # train in one "process": a best checkpoint is written to disk
    model = LogValLossModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", save_top_k=1, mode="min")
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=False,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)
    best_model_path = checkpoint_callback.best_model_path
    assert best_model_path
    assert os.path.exists(best_model_path)

    # simulate a fresh process: a new Trainer with a fresh ModelCheckpoint that has empty in-memory state
    fresh_checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", save_top_k=1, mode="min")
    assert fresh_checkpoint.best_model_path == ""
    fresh_trainer = Trainer(
        default_root_dir=tmp_path,
        limit_val_batches=2,
        limit_test_batches=2,
        limit_predict_batches=2,
        logger=False,
        callbacks=[fresh_checkpoint],
    )

    fn = getattr(fresh_trainer, trainer_fn)
    fn(LogValLossModel(), ckpt_path="best")
    assert os.path.normpath(fresh_trainer.ckpt_path) == os.path.normpath(best_model_path)


def test_best_ckpt_path_from_disk_on_remote_filesystem(tmp_path):
    """``ckpt_path="best"`` must be recoverable when checkpoints live on a non-local (fsspec) filesystem.

    Guards the disk-recovery path against assuming a local filesystem: candidates are opened through the callback's
    own ``self._fs`` (not ``torch.load`` on a raw path) and the recovered path is not ``normpath``-mangled, so
    ``memory://``/``s3://``-style paths work.

    """
    remote_dir = "memory://test_best_ckpt_remote_fs"
    remote_fs = get_filesystem(remote_dir)
    if remote_fs.exists(remote_dir):  # isolate from a previous run without clearing the shared memory store
        remote_fs.rm(remote_dir, recursive=True)

    class LogValLossModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            loss = self.step(batch)
            self.log("val_loss", loss)
            return {"x": loss}

    # train: a best checkpoint is written to the remote filesystem
    checkpoint_callback = ModelCheckpoint(dirpath=remote_dir, monitor="val_loss", save_top_k=1, mode="min")
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=False,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(LogValLossModel())
    best_model_path = checkpoint_callback.best_model_path
    assert best_model_path.startswith("memory:")
    assert get_filesystem(best_model_path).exists(best_model_path)

    # fresh process: empty in-memory state, must recover "best" from the remote filesystem
    fresh_checkpoint = ModelCheckpoint(dirpath=remote_dir, monitor="val_loss", save_top_k=1, mode="min")
    assert fresh_checkpoint.best_model_path == ""
    fresh_trainer = Trainer(default_root_dir=tmp_path, limit_val_batches=2, logger=False, callbacks=[fresh_checkpoint])
    fresh_trainer.validate(LogValLossModel(), ckpt_path="best")
    assert fresh_trainer.ckpt_path == best_model_path


def test_best_ckpt_path_no_disk_fallback_raises(tmp_path):
    """Test that ckpt_path="best" still raises a clear error when no checkpoint exists on disk."""
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", save_top_k=1, mode="min")
    trainer = Trainer(default_root_dir=tmp_path, logger=False, callbacks=[checkpoint_callback])
    with pytest.raises(ValueError, match="is not configured to save the best model"):
        trainer.validate(model, ckpt_path="best")


@pytest.mark.parametrize(("strict_loading", "expected"), [(None, True), (True, True), (False, False)])
def test_strict_loading(strict_loading, expected, tmp_path):
    """Test that the connector respects the `LightningModule.strict_loading` setting."""
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, barebones=True, max_steps=1)
    trainer.fit(model)
    trainer.save_checkpoint(tmp_path / "checkpoint.ckpt")

    model = BoringModel()
    model.strict_loading = strict_loading
    model.load_state_dict = Mock()

    trainer = Trainer(default_root_dir=tmp_path, barebones=True, max_steps=2)
    trainer.fit(model, ckpt_path=(tmp_path / "checkpoint.ckpt"))
    model.load_state_dict.assert_called_once_with(ANY, strict=expected)


@pytest.mark.parametrize("trainer_fn", ["validate", "test", "predict"])
def test_restore_callbacks_in_non_fit_phases(tmp_path, trainer_fn):
    """Test that callbacks are properly restored in non-fit phases."""

    class TestCallback(Callback):
        def __init__(self):
            self.restored = False

        def on_load_checkpoint(self, trainer, pl_module, checkpoint):
            if "callbacks" in checkpoint:
                callback_state = checkpoint["callbacks"][self.__class__.__name__]
                self.restored = callback_state["restored"]

        def state_dict(self):
            return {"restored": self.restored}

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint["callbacks"] = checkpoint.get("callbacks", {})
            checkpoint["callbacks"][self.__class__.__name__] = self.state_dict()

    # First create and train a model with the callback
    callback = TestCallback()
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmp_path, callbacks=[callback], max_steps=1)
    trainer.fit(model)

    # Set the callback state to True before saving
    callback.restored = True
    ckpt_path = tmp_path / "checkpoint.ckpt"
    trainer.save_checkpoint(ckpt_path)

    # Now create new instances and test restoration
    new_callback = TestCallback()
    new_model = BoringModel()
    assert not new_callback.restored  # Should start False

    new_trainer = Trainer(default_root_dir=tmp_path, callbacks=[new_callback])

    # Connect the model and restore callbacks before evaluation
    new_trainer.strategy.connect(new_model)
    new_trainer._checkpoint_connector.resume_start(ckpt_path)
    new_trainer._checkpoint_connector.restore_callbacks()

    # Run the evaluation phase (validate/test/predict)
    fn = getattr(new_trainer, trainer_fn)
    fn(new_model, ckpt_path=ckpt_path)

    assert new_callback.restored  # Should be True after loading the checkpoint
