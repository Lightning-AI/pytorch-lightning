# Copyright The PyTorch Lightning team.
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
import math
import os
import pickle
import re
import time
from argparse import Namespace
from datetime import timedelta
from logging import INFO
from pathlib import Path
from typing import Union
from unittest import mock
from unittest.mock import call, MagicMock, Mock, patch

import cloudpickle
import pytest
import torch
import yaml
from torch import optim

import pytorch_lightning as pl
import tests.helpers.utils as tutils
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _OMEGACONF_AVAILABLE
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf

if _OMEGACONF_AVAILABLE:
    from omegaconf import Container, OmegaConf


def test_model_checkpoint_state_key():
    early_stopping = ModelCheckpoint(monitor="val_loss")
    expected_id = (
        "ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
        " 'train_time_interval': None, 'save_on_train_epoch_end': None}"
    )
    assert early_stopping.state_key == expected_id


class LogInTwoMethods(BoringModel):
    def training_step(self, batch, batch_idx):
        out = super().training_step(batch, batch_idx)
        self.log("early_stop_on", out["loss"])
        return out

    def validation_epoch_end(self, outputs):
        outs = torch.stack([x["x"] for x in outputs]).mean()
        self.log("val_acc", outs)


def mock_training_epoch_loop(trainer):
    # do not use `unittest.Mock` because we need to store the return value
    calls = {}
    old_get_monitor_value = trainer.fit_loop.epoch_loop._get_monitor_value

    def mock(key):
        value = old_get_monitor_value(key)
        calls[trainer.current_epoch] = {key: value}
        return value

    trainer.fit_loop.epoch_loop._get_monitor_value = mock
    return calls


@pytest.mark.parametrize(
    "validation_step_none,val_dataloaders_none,monitor",
    [(False, False, "val_log"), (True, False, "train_log_epoch"), (False, True, "val_log")],
)
@pytest.mark.parametrize("reduce_lr_on_plateau", [False, True])
def test_model_checkpoint_score_and_ckpt(
    tmpdir, validation_step_none: bool, val_dataloaders_none: bool, monitor: str, reduce_lr_on_plateau: bool
):
    """Test that when a model checkpoint is saved, it saves with the correct score appended to ckpt_path and
    checkpoint data."""
    max_epochs = 3
    limit_train_batches = 5
    limit_val_batches = 7
    lr, gamma = 1e-1, 2

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.train_log_epochs = torch.randn(max_epochs, limit_train_batches)
            self.val_logs = torch.randn(max_epochs, limit_val_batches)
            self.scores = []

        def training_step(self, batch, batch_idx):
            log_value = self.train_log_epochs[self.current_epoch, batch_idx]
            self.log("train_log", log_value, on_epoch=True)
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            log_value = self.val_logs[self.current_epoch, batch_idx]
            self.log("val_log", log_value)
            self.log("epoch", self.current_epoch, on_epoch=True)
            return super().validation_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer = optim.SGD(self.parameters(), lr=lr)

            if reduce_lr_on_plateau:
                lr_scheduler = {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                    "monitor": monitor,
                    "strict": True,
                }
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

            return [optimizer], [lr_scheduler]

        def on_train_epoch_end(self):
            if "train" in monitor:
                self.scores.append(self.trainer.logged_metrics[monitor])

        def on_validation_epoch_end(self):
            if not self.trainer.sanity_checking and "val" in monitor:
                self.scores.append(self.trainer.logged_metrics[monitor])

    filename = "{" + f"{monitor}" + ":.4f}-{epoch}"
    checkpoint = ModelCheckpoint(dirpath=tmpdir, filename=filename, monitor=monitor, save_top_k=-1)

    model = CustomBoringModel()

    if validation_step_none:
        model.validation_step = None
    if val_dataloaders_none:
        model.val_dataloaders = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=max_epochs,
        enable_progress_bar=False,
    )
    calls = mock_training_epoch_loop(trainer)
    trainer.fit(model)

    ckpt_files = list(Path(tmpdir).glob("*.ckpt"))
    assert len(ckpt_files) == len(model.scores) == max_epochs

    for epoch in range(max_epochs):
        score = model.scores[epoch]
        expected_score = getattr(model, f"{monitor}s")[epoch].mean().item()
        assert math.isclose(score, expected_score, rel_tol=1e-4)

        expected_filename = f"{monitor}={score:.4f}-epoch={epoch}.ckpt"
        chk = pl_load(os.path.join(checkpoint.dirpath, expected_filename))
        assert chk["epoch"] == epoch
        assert chk["global_step"] == limit_train_batches * (epoch + 1)

        mc_specific_data = chk["callbacks"][
            f"ModelCheckpoint{{'monitor': '{monitor}', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
            " 'train_time_interval': None, 'save_on_train_epoch_end': True}"
        ]
        assert mc_specific_data["dirpath"] == checkpoint.dirpath
        assert mc_specific_data["monitor"] == monitor
        assert mc_specific_data["current_score"] == score

        if not reduce_lr_on_plateau:
            actual_step_count = chk["lr_schedulers"][0]["_step_count"]
            actual_lr = chk["lr_schedulers"][0]["_last_lr"][0]
            # checkpoint is saved after updating lr_scheduler states
            assert actual_step_count == epoch + 2  # step_count starts at 1
            assert actual_lr == lr * gamma ** (epoch + 1)
        else:
            assert calls[epoch] == {monitor: score}


@pytest.mark.parametrize(
    "val_check_interval,reduce_lr_on_plateau,epoch_aligned",
    [(0.25, True, True), (0.25, False, True), (0.42, False, False)],
)
def test_model_checkpoint_score_and_ckpt_val_check_interval(
    tmpdir, val_check_interval, reduce_lr_on_plateau, epoch_aligned
):
    """Test that when a model checkpoint is saved, it saves with the correct score appended to ckpt_path and
    checkpoint data with val_check_interval."""
    seed_everything(0)
    max_epochs = 3
    limit_train_batches = 12
    limit_val_batches = 7
    lr, gamma = 1e-1, 2
    monitor = "val_log"
    per_val_train_batches = int(limit_train_batches * val_check_interval)
    per_epoch_val_checks, leftover_train_batches = divmod(limit_train_batches, per_val_train_batches)

    class CustomBoringModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.val_logs = torch.randn(per_epoch_val_checks * max_epochs, limit_val_batches)
            self.val_loop_count = 0
            self.scores = []

        def validation_step(self, batch, batch_idx):
            log_value = self.val_logs[self.val_loop_count, batch_idx]
            self.log("val_log", log_value)
            return super().validation_step(batch, batch_idx)

        def validation_epoch_end(self, outputs):
            self.val_loop_count += 1
            super().validation_epoch_end(outputs)
            self.scores.append(self.trainer.logged_metrics[monitor])

        def configure_optimizers(self):
            optimizer = optim.SGD(self.parameters(), lr=lr)

            if reduce_lr_on_plateau:
                lr_scheduler = {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                    "monitor": monitor,
                    "strict": True,
                }
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

            return [optimizer], [lr_scheduler]

    filename = "{" + f"{monitor}" + ":.4f}-{epoch}"
    checkpoint = ModelCheckpoint(dirpath=tmpdir, filename=filename, monitor=monitor, save_top_k=-1)

    model = CustomBoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    calls = mock_training_epoch_loop(trainer)
    trainer.fit(model)

    def _make_assertions(epoch, ix):
        global_ix = ix + per_epoch_val_checks * epoch

        # checkpoint saved at the end of training epoch will have updated lr_scheduler states
        epoch_end_checkpoint = epoch_aligned and ix == (per_epoch_val_checks - 1)

        score = model.scores[global_ix]
        expected_score = getattr(model, f"{monitor}s")[global_ix].mean().item()
        expected_filename = f"{monitor}={score:.4f}-epoch={epoch}.ckpt"
        assert math.isclose(score, expected_score, rel_tol=1e-4)

        chk = pl_load(os.path.join(checkpoint.dirpath, expected_filename))
        assert chk["epoch"] == epoch
        expected_global_step = per_val_train_batches * (global_ix + 1) + (leftover_train_batches * epoch)
        assert chk["global_step"] == expected_global_step

        mc_specific_data = chk["callbacks"][
            f"ModelCheckpoint{{'monitor': '{monitor}', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
            " 'train_time_interval': None, 'save_on_train_epoch_end': False}"
        ]
        assert mc_specific_data["dirpath"] == checkpoint.dirpath
        assert mc_specific_data["monitor"] == monitor
        assert mc_specific_data["current_score"] == score

        if not reduce_lr_on_plateau:
            actual_step_count = chk["lr_schedulers"][0]["_step_count"]
            actual_lr = chk["lr_schedulers"][0]["_last_lr"][0]
            assert actual_step_count == epoch + 1 + epoch_end_checkpoint
            assert actual_lr == lr * gamma ** (epoch + epoch_end_checkpoint)

        return score

    ckpt_files = list(Path(tmpdir).glob("*.ckpt"))
    assert len(ckpt_files) == len(model.scores) == per_epoch_val_checks * max_epochs

    for epoch in range(max_epochs):
        for i in range(per_epoch_val_checks):
            score = _make_assertions(epoch, i)

        if reduce_lr_on_plateau:
            assert calls[epoch] == {monitor: score}


@pytest.mark.parametrize("save_top_k", [-1, 0, 1, 2])
def test_model_checkpoint_with_non_string_input(tmpdir, save_top_k: int):
    """Test that dirpath=None in checkpoint callback is valid and that ckpt_path is set correctly."""
    tutils.reset_seed()
    model = LogInTwoMethods()

    checkpoint = ModelCheckpoint(monitor="early_stop_on", dirpath=None, filename="{epoch}", save_top_k=save_top_k)
    max_epochs = 2
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint], overfit_batches=0.20, max_epochs=max_epochs)
    trainer.fit(model)
    assert checkpoint.dirpath == tmpdir / trainer.logger.name / "version_0" / "checkpoints"

    if save_top_k == -1:
        ckpt_files = os.listdir(checkpoint.dirpath)
        expected_ckpt_files = [f"epoch={i}.ckpt" for i in range(max_epochs)]
        assert len(ckpt_files) == len(expected_ckpt_files) == max_epochs
        assert set(ckpt_files) == set(expected_ckpt_files)


@pytest.mark.parametrize("save_top_k", [-1, 0, 1, 2])
def test_model_checkpoint_to_yaml(tmpdir, save_top_k: int):
    """Test that None in checkpoint callback is valid and that chkp_path is set correctly."""
    tutils.reset_seed()
    model = LogInTwoMethods()

    checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor="early_stop_on", save_top_k=save_top_k)

    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint], overfit_batches=0.20, max_epochs=2)
    trainer.fit(model)

    path_yaml = os.path.join(tmpdir, "best_k_models.yaml")
    checkpoint.to_yaml(path_yaml)
    d = yaml.full_load(open(path_yaml))
    best_k = dict(checkpoint.best_k_models.items())
    assert d == best_k


@pytest.mark.parametrize("logger_version,expected", [(None, "version_0"), (1, "version_1"), ("awesome", "awesome")])
def test_model_checkpoint_path(tmpdir, logger_version: Union[None, int, str], expected: str):
    """Test that "version_" prefix is only added when logger's version is an integer."""
    tutils.reset_seed()
    model = LogInTwoMethods()
    logger = TensorBoardLogger(str(tmpdir), version=logger_version)

    trainer = Trainer(default_root_dir=tmpdir, overfit_batches=0.2, max_epochs=2, logger=logger)
    trainer.fit(model)

    ckpt_version = Path(trainer.checkpoint_callback.dirpath).parent.name
    assert ckpt_version == expected


def test_pickling(tmpdir):
    ckpt = ModelCheckpoint(dirpath=tmpdir)

    ckpt_pickled = pickle.dumps(ckpt)
    ckpt_loaded = pickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)

    ckpt_pickled = cloudpickle.dumps(ckpt)
    ckpt_loaded = cloudpickle.loads(ckpt_pickled)
    assert vars(ckpt) == vars(ckpt_loaded)


class ModelCheckpointTestInvocations(ModelCheckpoint):
    # this class has to be defined outside the test function, otherwise we get pickle error
    # due to the way ddp process is launched

    def __init__(self, expected_count, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_count = expected_count
        self.on_save_checkpoint_count = 0

    def on_train_start(self, trainer, pl_module):
        torch.save = Mock(wraps=torch.save)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # only rank 0 will call ``torch.save``
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        self.on_save_checkpoint_count += 1

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        assert self.best_model_path
        assert self.best_model_score
        assert self.on_save_checkpoint_count == self.expected_count
        if trainer.is_global_zero:
            assert torch.save.call_count == self.expected_count
        else:
            assert torch.save.call_count == 0


@RunIf(skip_windows=True, skip_49370=True)
def test_model_checkpoint_no_extraneous_invocations(tmpdir):
    """Test to ensure that the model callback saves the checkpoints only once in distributed mode."""
    model = LogInTwoMethods()
    num_epochs = 4
    model_checkpoint = ModelCheckpointTestInvocations(monitor="early_stop_on", expected_count=num_epochs, save_top_k=-1)
    trainer = Trainer(
        strategy="ddp_spawn",
        accelerator="cpu",
        devices=2,
        default_root_dir=tmpdir,
        callbacks=[model_checkpoint],
        max_epochs=num_epochs,
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


def test_model_checkpoint_format_checkpoint_name(tmpdir):
    # empty filename:
    ckpt_name = ModelCheckpoint._format_checkpoint_name("", {"epoch": 3, "step": 2})
    assert ckpt_name == "epoch=3-step=2"

    ckpt_name = ModelCheckpoint._format_checkpoint_name(None, {"epoch": 3, "step": 2}, prefix="test")
    assert ckpt_name == "test-epoch=3-step=2"

    # no groups case:
    ckpt_name = ModelCheckpoint._format_checkpoint_name("ckpt", {}, prefix="test")
    assert ckpt_name == "test-ckpt"

    # no prefix
    ckpt_name = ModelCheckpoint._format_checkpoint_name("{epoch:03d}-{acc}", {"epoch": 3, "acc": 0.03})
    assert ckpt_name == "epoch=003-acc=0.03"

    # prefix
    char_org = ModelCheckpoint.CHECKPOINT_JOIN_CHAR
    ModelCheckpoint.CHECKPOINT_JOIN_CHAR = "@"
    ckpt_name = ModelCheckpoint._format_checkpoint_name("{epoch},{acc:.5f}", {"epoch": 3, "acc": 0.03}, prefix="test")
    assert ckpt_name == "test@epoch=3,acc=0.03000"
    ModelCheckpoint.CHECKPOINT_JOIN_CHAR = char_org

    # no dirpath set
    ckpt_name = ModelCheckpoint(monitor="early_stop_on", dirpath=None).format_checkpoint_name({"epoch": 3, "step": 2})
    assert ckpt_name == "epoch=3-step=2.ckpt"
    ckpt_name = ModelCheckpoint(monitor="early_stop_on", dirpath="").format_checkpoint_name({"epoch": 5, "step": 4})
    assert ckpt_name == "epoch=5-step=4.ckpt"

    # CWD
    ckpt_name = ModelCheckpoint(monitor="early_stop_on", dirpath=".").format_checkpoint_name({"epoch": 3, "step": 4})
    assert ckpt_name == str(Path(".").resolve() / "epoch=3-step=4.ckpt")

    # with version
    ckpt = ModelCheckpoint(monitor="early_stop_on", dirpath=tmpdir, filename="name")
    ckpt_name = ckpt.format_checkpoint_name({}, ver=3)
    assert ckpt_name == tmpdir / "name-v3.ckpt"

    # using slashes
    ckpt = ModelCheckpoint(monitor="early_stop_on", dirpath=None, filename="{epoch}_{val/loss:.5f}")
    ckpt_name = ckpt.format_checkpoint_name({"epoch": 4, "val/loss": 0.03})
    assert ckpt_name == "epoch=4_val/loss=0.03000.ckpt"

    # auto_insert_metric_name=False
    ckpt_name = ModelCheckpoint._format_checkpoint_name(
        "epoch={epoch:03d}-val_acc={val/acc}", {"epoch": 3, "val/acc": 0.03}, auto_insert_metric_name=False
    )
    assert ckpt_name == "epoch=003-val_acc=0.03"


class ModelCheckpointExtensionTest(ModelCheckpoint):
    FILE_EXTENSION = ".tpkc"


def test_model_checkpoint_file_extension(tmpdir):
    """Test ModelCheckpoint with different file extension."""
    model = LogInTwoMethods()
    model_checkpoint = ModelCheckpointExtensionTest(
        monitor="early_stop_on", dirpath=tmpdir, save_top_k=1, save_last=True
    )
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[model_checkpoint], max_steps=1, logger=False)
    trainer.fit(model)

    expected = ["epoch=0-step=1.tpkc", "last.tpkc"]
    assert set(expected) == set(os.listdir(tmpdir))


def test_model_checkpoint_save_last(tmpdir):
    """Tests that save_last produces only one last checkpoint."""
    seed_everything()
    model = LogInTwoMethods()
    epochs = 3
    ModelCheckpoint.CHECKPOINT_NAME_LAST = "last-{epoch}"
    model_checkpoint = ModelCheckpoint(monitor="early_stop_on", dirpath=tmpdir, save_top_k=-1, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[model_checkpoint],
        max_epochs=epochs,
        limit_train_batches=10,
        limit_val_batches=10,
        logger=False,
    )
    trainer.fit(model)
    last_filename = model_checkpoint._format_checkpoint_name(
        ModelCheckpoint.CHECKPOINT_NAME_LAST, {"epoch": trainer.current_epoch - 1}
    )
    last_filename = last_filename + ".ckpt"
    assert str(tmpdir / last_filename) == model_checkpoint.last_model_path
    assert set(os.listdir(tmpdir)) == set(
        [f"epoch={i}-step={j}.ckpt" for i, j in zip(range(epochs), [10, 20, 30])] + [last_filename]
    )

    ModelCheckpoint.CHECKPOINT_NAME_LAST = "last"


def test_invalid_top_k(tmpdir):
    """Make sure that a MisconfigurationException is raised for a negative save_top_k argument."""
    with pytest.raises(MisconfigurationException, match=r".*Must be >= -1"):
        ModelCheckpoint(dirpath=tmpdir, save_top_k=-3)


def test_none_monitor_top_k(tmpdir):
    """Test that a warning appears for positive top_k with monitor=None."""
    with pytest.raises(
        MisconfigurationException, match=r"ModelCheckpoint\(save_top_k=3, monitor=None\) is not a valid*"
    ):
        ModelCheckpoint(dirpath=tmpdir, save_top_k=3)
    # These should not fail
    ModelCheckpoint(dirpath=tmpdir, save_top_k=-1)
    ModelCheckpoint(dirpath=tmpdir, save_top_k=0)
    ModelCheckpoint(dirpath=tmpdir, save_top_k=1)


def test_invalid_every_n_epochs(tmpdir):
    """Make sure that a MisconfigurationException is raised for a negative every_n_epochs argument."""
    with pytest.raises(MisconfigurationException, match=r".*Must be >= 0"):
        ModelCheckpoint(dirpath=tmpdir, every_n_epochs=-3)
    # These should not fail
    ModelCheckpoint(dirpath=tmpdir, every_n_epochs=0)
    ModelCheckpoint(dirpath=tmpdir, every_n_epochs=1)
    ModelCheckpoint(dirpath=tmpdir, every_n_epochs=2)


def test_invalid_every_n_train_steps(tmpdir):
    """Make sure that a MisconfigurationException is raised for a negative every_n_epochs argument."""
    with pytest.raises(MisconfigurationException, match=r".*Must be >= 0"):
        ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=-3)
    # These should not fail
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=0)
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=1)
    ModelCheckpoint(dirpath=tmpdir, every_n_epochs=2)


def test_invalid_trigger_combination(tmpdir):
    """Test that a MisconfigurationException is raised if more than one of every_n_epochs, every_n_train_steps, and
    train_time_interval are enabled together."""
    with pytest.raises(MisconfigurationException, match=r".*Combination of parameters every_n_train_steps"):
        ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=1, every_n_epochs=2)
    with pytest.raises(MisconfigurationException, match=r".*Combination of parameters every_n_train_steps"):
        ModelCheckpoint(train_time_interval=timedelta(minutes=1), every_n_epochs=2)
    with pytest.raises(MisconfigurationException, match=r".*Combination of parameters every_n_train_steps"):
        ModelCheckpoint(train_time_interval=timedelta(minutes=1), every_n_train_steps=2)

    # These should not fail
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=0, every_n_epochs=3)
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=4, every_n_epochs=0)
    ModelCheckpoint(dirpath=tmpdir, every_n_train_steps=0, every_n_epochs=0, train_time_interval=timedelta(minutes=1))


def test_none_every_n_train_steps_val_epochs(tmpdir):
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir)
    assert checkpoint_callback.every_n_epochs == 1
    assert checkpoint_callback._every_n_train_steps == 0


def test_model_checkpoint_save_last_none_monitor(tmpdir, caplog):
    """Test that it is possible to save all checkpoints when monitor=None."""
    seed_everything()
    model = LogInTwoMethods()

    epochs = 2
    checkpoint_callback = ModelCheckpoint(monitor=None, dirpath=tmpdir, save_top_k=-1, save_last=True)
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint_callback],
        limit_train_batches=10,
        limit_val_batches=10,
        max_epochs=epochs,
        logger=False,
    )

    with caplog.at_level(INFO):
        trainer.fit(model)
        assert "will duplicate the last checkpoint saved" in caplog.text

    # these should not be set if monitor is None
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == tmpdir / "epoch=1-step=20.ckpt"
    assert checkpoint_callback.last_model_path == tmpdir / "last.ckpt"
    assert checkpoint_callback.best_model_score is None
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ""

    # check that the correct ckpts were created
    expected = [f"epoch={i}-step={j}.ckpt" for i, j in zip(range(epochs), [10, 20])]
    expected.append("last.ckpt")
    assert set(os.listdir(tmpdir)) == set(expected)


@pytest.mark.parametrize("every_n_epochs", list(range(4)))
def test_model_checkpoint_every_n_epochs(tmpdir, every_n_epochs):
    model = LogInTwoMethods()
    epochs = 5
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmpdir, filename="{epoch}", save_top_k=-1, every_n_epochs=every_n_epochs
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint_callback],
        max_epochs=epochs,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
    )
    trainer.fit(model)

    # check that the correct ckpts were created, the modulo condition is checked in `ModelCheckpoint`
    expected = [f"epoch={e}.ckpt" for e in range(epochs) if not (e + 1) % every_n_epochs] if every_n_epochs > 0 else []
    assert set(os.listdir(tmpdir)) == set(expected)


def test_ckpt_every_n_train_steps(tmpdir):
    """Tests that the checkpoints are saved every n training steps."""

    model = LogInTwoMethods()
    every_n_train_steps = 16
    max_epochs = 2
    epoch_length = 64
    checkpoint_callback = ModelCheckpoint(
        filename="{step}",
        every_n_epochs=0,
        every_n_train_steps=every_n_train_steps,
        dirpath=tmpdir,
        save_top_k=-1,
        save_last=False,
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback],
        logger=False,
    )

    trainer.fit(model)
    expected = [
        f"step={i}.ckpt" for i in range(every_n_train_steps, max_epochs * epoch_length + 1, every_n_train_steps)
    ]
    assert set(os.listdir(tmpdir)) == set(expected)


@mock.patch("pytorch_lightning.callbacks.model_checkpoint.time")
def test_model_checkpoint_train_time_interval(mock_datetime, tmpdir) -> None:
    """Tests that the checkpoints are saved at the specified time interval."""
    seconds_per_batch = 7
    start_time = time.monotonic()
    batches_per_epoch = 64
    num_epochs = 2
    max_batches = batches_per_epoch * num_epochs + 1
    mock_datetime.monotonic.side_effect = [start_time + seconds_per_batch * i for i in range(max_batches)]

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        min_epochs=num_epochs,
        max_epochs=num_epochs,
        enable_progress_bar=False,
        callbacks=[
            ModelCheckpoint(
                filename="{epoch}-{step}",
                dirpath=tmpdir,
                train_time_interval=timedelta(minutes=1),
                save_top_k=-1,
                save_last=False,
            )
        ],
        logger=False,
    )

    trainer.fit(model)
    # Each batch takes 7 sec and we checkpoint every minute. There are 64
    # batches per epoch, so total time to run is 7*64*2 = 896 sec < 14.96 minutes,
    # so we should have 14 checkpoints.
    assert len(os.listdir(tmpdir)) == 14


def test_model_checkpoint_topk_zero(tmpdir):
    """Test that no checkpoints are saved when save_top_k=0."""
    model = LogInTwoMethods()
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=0, save_last=True)
    trainer = Trainer(default_root_dir=tmpdir, callbacks=[checkpoint_callback], max_epochs=2, logger=False)
    trainer.fit(model)
    # these should not be set if monitor is None
    assert checkpoint_callback.monitor is None
    assert checkpoint_callback.best_model_path == ""
    assert checkpoint_callback.best_model_score is None
    assert checkpoint_callback.best_k_models == {}
    assert checkpoint_callback.kth_best_model_path == ""
    # check that only the last ckpt was created
    assert os.listdir(tmpdir) == ["last.ckpt"]
    assert checkpoint_callback.last_model_path == tmpdir / "last.ckpt"


def test_model_checkpoint_topk_all(tmpdir):
    """Test that save_top_k=-1 tracks the best models when monitor key is provided."""
    seed_everything(1000)
    epochs = 3

    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(
        dirpath=tmpdir, filename="{epoch}", monitor="epoch", mode="max", save_top_k=-1
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[checkpoint_callback],
        max_epochs=epochs,
        logger=False,
        val_check_interval=1.0,
    )
    trainer.fit(model)

    assert checkpoint_callback.monitor == "epoch"
    assert checkpoint_callback.best_model_path == tmpdir / "epoch=2.ckpt"
    assert checkpoint_callback.best_model_score == epochs - 1
    assert len(os.listdir(tmpdir)) == len(checkpoint_callback.best_k_models) == epochs
    assert set(checkpoint_callback.best_k_models.keys()) == {str(tmpdir / f"epoch={i}.ckpt") for i in range(epochs)}
    assert checkpoint_callback.kth_best_model_path == tmpdir / "epoch=0.ckpt"


def test_ckpt_metric_names(tmpdir):
    model = LogInTwoMethods()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gradient_clip_val=1.0,
        overfit_batches=0.20,
        enable_progress_bar=False,
        limit_train_batches=0.01,
        limit_val_batches=0.01,
        callbacks=[ModelCheckpoint(monitor="early_stop_on", dirpath=tmpdir, filename="{val_loss:.2f}")],
    )

    trainer.fit(model)

    # make sure the checkpoint we saved has the metric in the name
    ckpts = os.listdir(tmpdir)
    ckpts = [x for x in ckpts if "val_loss" in x]
    assert len(ckpts) == 1
    val = re.sub("[^0-9.]", "", ckpts[0])
    assert len(val) > 3


def test_default_checkpoint_behavior(tmpdir):
    seed_everything(1234)

    model = LogInTwoMethods()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=3, enable_progress_bar=False, limit_train_batches=5, limit_val_batches=5
    )

    with patch.object(trainer, "save_checkpoint", wraps=trainer.save_checkpoint) as save_mock:
        trainer.fit(model)
        results = trainer.test()

    assert len(results) == 1
    save_dir = tmpdir / "lightning_logs" / "version_0" / "checkpoints"
    save_weights_only = trainer.checkpoint_callback.save_weights_only
    save_mock.assert_has_calls(
        [
            call(save_dir / "epoch=0-step=5.ckpt", save_weights_only),
            call(save_dir / "epoch=1-step=10.ckpt", save_weights_only),
            call(save_dir / "epoch=2-step=15.ckpt", save_weights_only),
        ]
    )
    ckpts = os.listdir(save_dir)
    assert len(ckpts) == 1
    assert ckpts[0] == "epoch=2-step=15.ckpt"


def test_model_checkpoint_save_last_checkpoint_contents(tmpdir):
    """Tests that the save_last checkpoint contains the latest information."""
    seed_everything(100)
    model = LogInTwoMethods()
    num_epochs = 3
    model_checkpoint = ModelCheckpoint(
        monitor="early_stop_on", dirpath=tmpdir, filename="{epoch}", save_top_k=num_epochs, save_last=True
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[model_checkpoint],
        max_epochs=num_epochs,
        limit_train_batches=2,
        limit_val_batches=2,
    )
    trainer.fit(model)

    path_last_epoch = str(tmpdir / f"epoch={num_epochs - 1}.ckpt")
    path_last = str(tmpdir / "last.ckpt")
    assert path_last == model_checkpoint.last_model_path
    assert os.path.isfile(path_last_epoch)

    ckpt_last_epoch = torch.load(path_last_epoch)
    ckpt_last = torch.load(path_last)

    assert ckpt_last_epoch["epoch"] == ckpt_last["epoch"]
    assert ckpt_last_epoch["global_step"] == ckpt_last["global_step"]

    ckpt_id = (
        "ModelCheckpoint{'monitor': 'early_stop_on', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
        " 'train_time_interval': None, 'save_on_train_epoch_end': True}"
    )
    assert ckpt_last["callbacks"][ckpt_id] == ckpt_last_epoch["callbacks"][ckpt_id]

    # it is easier to load the model objects than to iterate over the raw dict of tensors
    model_last_epoch = LogInTwoMethods.load_from_checkpoint(path_last_epoch)
    model_last = LogInTwoMethods.load_from_checkpoint(model_checkpoint.last_model_path)
    for w0, w1 in zip(model_last_epoch.parameters(), model_last.parameters()):
        assert w0.eq(w1).all()


@pytest.mark.parametrize("mode", ["min", "max"])
def test_checkpointing_with_nan_as_first(tmpdir, mode):
    monitor = [float("nan")]
    monitor += [5, 7, 8] if mode == "max" else [8, 7, 5]

    class CurrentModel(LogInTwoMethods):
        def validation_epoch_end(self, outputs):
            val_loss = monitor[self.current_epoch]
            self.log("abc", val_loss)

    model = CurrentModel()

    callback = ModelCheckpoint(monitor="abc", mode=mode, save_top_k=1, dirpath=tmpdir)

    trainer = Trainer(
        callbacks=[callback],
        default_root_dir=tmpdir,
        val_check_interval=1.0,
        max_epochs=len(monitor),
    )
    trainer.save_checkpoint = MagicMock()

    trainer.fit(model)

    # check that last one is also the best one
    assert trainer.save_checkpoint.call_count == len(monitor)
    assert mode == "min" and callback.best_model_score == 5 or mode == "max" and callback.best_model_score == 8


def test_checkpoint_repeated_strategy(tmpdir):
    """This test validates checkpoint can be called several times without increasing internally its global step if
    nothing run."""
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=tmpdir, filename="{epoch:02d}")

    class ExtendedBoringModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            self.log("val_loss", loss)

    model = ExtendedBoringModel()
    model.validation_epoch_end = None
    trainer_kwargs = {
        "max_epochs": 1,
        "limit_train_batches": 2,
        "limit_val_batches": 2,
        "limit_test_batches": 2,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    trainer = Trainer(**trainer_kwargs, callbacks=[checkpoint_callback])
    trainer.fit(model)
    assert os.listdir(tmpdir) == ["epoch=00.ckpt"]

    for idx in range(4):
        # load from checkpoint
        trainer = pl.Trainer(**trainer_kwargs, default_root_dir=tmpdir)
        trainer.fit(model, ckpt_path=checkpoint_callback.best_model_path)
        trainer.test(ckpt_path=checkpoint_callback.best_model_path, verbose=False)
        assert set(os.listdir(tmpdir)) == {"epoch=00.ckpt", "lightning_logs"}
    assert set(os.listdir(tmpdir / "lightning_logs")) == {f"version_{i}" for i in range(4)}


def test_checkpoint_repeated_strategy_extended(tmpdir):
    """This test validates checkpoint can be called several times without increasing internally its global step if
    nothing run."""

    class ExtendedBoringModel(BoringModel):
        def validation_step(self, batch, batch_idx):
            output = self.layer(batch)
            loss = self.loss(batch, output)
            return {"val_loss": loss}

        def validation_epoch_end(self, *_):
            ...

    def assert_trainer_init(trainer):
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def get_last_checkpoint(ckpt_dir):
        last = ckpt_dir.listdir(sort=True)[-1]
        return str(last)

    def assert_checkpoint_content(ckpt_dir):
        chk = pl_load(get_last_checkpoint(ckpt_dir))
        # `-1` because this checkpoint is saved `on_train_epoch_end` which is considered part of the epoch so the
        # `current_epoch` count has not been increased yet
        assert chk["epoch"] == epochs - 1
        assert chk["global_step"] == 4

    def assert_checkpoint_log_dir(idx):
        lightning_logs = tmpdir / "lightning_logs"
        actual = [d.basename for d in lightning_logs.listdir(sort=True)]
        assert actual == [f"version_{i}" for i in range(idx + 1)]
        actual = [d.basename for d in ckpt_dir.listdir()]
        assert len(actual) == epochs, actual

    ckpt_dir = tmpdir / "checkpoints"
    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1)
    epochs = 2
    limit_train_batches = 2
    trainer_config = dict(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=3,
        limit_test_batches=4,
        callbacks=[checkpoint_cb],
    )
    trainer = pl.Trainer(**trainer_config)
    assert_trainer_init(trainer)

    model = ExtendedBoringModel()
    trainer.fit(model)
    assert trainer.global_step == epochs * limit_train_batches
    assert trainer.current_epoch == epochs
    assert_checkpoint_log_dir(0)
    assert_checkpoint_content(ckpt_dir)

    trainer.validate(model)
    assert trainer.current_epoch == epochs

    trainer.test(model)
    assert trainer.current_epoch == epochs

    for idx in range(1, 5):
        chk = get_last_checkpoint(ckpt_dir)
        assert_checkpoint_content(ckpt_dir)

        # load from checkpoint
        trainer = pl.Trainer(**trainer_config)
        assert_trainer_init(trainer)

        model = ExtendedBoringModel()

        trainer.test(model)
        assert_trainer_init(trainer)

        trainer.fit(model, ckpt_path=chk)
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert trainer.fit_loop.epoch_progress.current.processed == epochs

        trainer.validate(model)
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert trainer.fit_loop.epoch_progress.current.processed == epochs

        trainer.fit(model)
        assert trainer.global_step == epochs * limit_train_batches
        assert trainer.current_epoch == epochs
        assert trainer.fit_loop.epoch_progress.current.processed == epochs
        assert_checkpoint_log_dir(idx)


def test_configure_model_checkpoint(tmpdir):
    """Test all valid and invalid ways a checkpoint callback can be passed to the Trainer."""
    kwargs = dict(default_root_dir=tmpdir)
    callback1 = ModelCheckpoint()
    callback2 = ModelCheckpoint()

    # no callbacks
    trainer = Trainer(enable_checkpointing=False, callbacks=[], **kwargs)
    assert not any(isinstance(c, ModelCheckpoint) for c in trainer.callbacks)
    assert trainer.checkpoint_callback is None

    # default configuration
    trainer = Trainer(callbacks=[], **kwargs)
    assert sum(1 for c in trainer.callbacks if isinstance(c, ModelCheckpoint)) == 1
    assert isinstance(trainer.checkpoint_callback, ModelCheckpoint)

    # custom callback passed to callbacks list, enable_checkpointing=True is ignored
    trainer = Trainer(enable_checkpointing=True, callbacks=[callback1], **kwargs)
    assert [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)] == [callback1]
    assert trainer.checkpoint_callback == callback1

    # multiple checkpoint callbacks
    trainer = Trainer(callbacks=[callback1, callback2], **kwargs)
    assert trainer.checkpoint_callback == callback1
    assert trainer.checkpoint_callbacks == [callback1, callback2]

    with pytest.raises(MisconfigurationException, match="`enable_checkpointing=False` but found `ModelCheckpoint`"):
        Trainer(enable_checkpointing=False, callbacks=[callback1], **kwargs)


def test_val_check_interval_checkpoint_files(tmpdir):
    """Test correct checkpoint naming when validating/checkpointing multiple times per epoch."""
    model = LogInTwoMethods()
    model_checkpoint = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, monitor="val_acc", mode="max")
    trainer = Trainer(
        default_root_dir=tmpdir,
        val_check_interval=0.2,
        max_epochs=1,
        limit_train_batches=10,
        callbacks=[model_checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    files = {p.basename for p in tmpdir.listdir()}
    assert files == {f"epoch=0-step={s}.ckpt" for s in [2, 4, 6, 8, 10]}


def test_current_score(tmpdir):
    """Check that the current_score value is correct and was saved."""

    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", (self.current_epoch + 1) / 10)
            return super().training_step(*args)

    model_checkpoint = ModelCheckpoint(dirpath=tmpdir, save_top_k=3, monitor="foo", mode="min")
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=3,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[model_checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(TestModel())
    assert model_checkpoint.current_score == 0.3
    ckpts = [torch.load(str(ckpt)) for ckpt in tmpdir.listdir()]
    ckpts = [
        ckpt["callbacks"][
            "ModelCheckpoint{'monitor': 'foo', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1,"
            " 'train_time_interval': None, 'save_on_train_epoch_end': True}"
        ]
        for ckpt in ckpts
    ]
    assert sorted(ckpt["current_score"] for ckpt in ckpts) == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("mode", ["min", "max"])
def test_current_score_when_nan(tmpdir, mode: str):
    """Check that ModelCheckpoint handles NaN values correctly."""

    class TestModel(BoringModel):
        def training_step(self, *args):
            self.log("foo", float("nan"))
            return super().training_step(*args)

    model_checkpoint = ModelCheckpoint(dirpath=tmpdir, save_top_k=1, monitor="foo", mode=mode)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[model_checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(TestModel())
    expected = float("inf" if mode == "min" else "-inf")
    assert model_checkpoint.best_model_score == expected
    assert model_checkpoint.current_score == expected


@pytest.mark.parametrize("use_omegaconf", [False, pytest.param(True, marks=RunIf(omegaconf=True))])
def test_hparams_type(tmpdir, use_omegaconf):
    class TestModel(BoringModel):
        def __init__(self, hparams):
            super().__init__()
            self.save_hyperparameters(hparams)

    model_checkpoint = ModelCheckpoint(dirpath=tmpdir, save_top_k=1, monitor="foo")
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=[model_checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    hp = {"test_hp_0": 1, "test_hp_1": 2}
    hp = OmegaConf.create(hp) if use_omegaconf else Namespace(**hp)
    model = TestModel(hp)
    trainer.fit(model)
    ckpt = trainer._checkpoint_connector.dump_checkpoint()
    if use_omegaconf:
        assert isinstance(ckpt[model.CHECKPOINT_HYPER_PARAMS_KEY], Container)
    else:
        # make sure it's not AttributeDict
        assert type(ckpt[model.CHECKPOINT_HYPER_PARAMS_KEY]) is dict


def test_ckpt_version_after_rerun_new_trainer(tmpdir):
    """Check that previous checkpoints are renamed to have the correct version suffix when new trainer instances
    are used."""
    epochs = 2
    for i in range(epochs):
        mc = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, monitor="epoch", filename="{epoch}")
        trainer = Trainer(
            max_epochs=epochs,
            limit_train_batches=1,
            limit_val_batches=1,
            default_root_dir=tmpdir,
            callbacks=[mc],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(BoringModel())

        # check best_k_models state
        expected = {"epoch=0-v1.ckpt", "epoch=1-v1.ckpt"} if i else {"epoch=0.ckpt", "epoch=1.ckpt"}
        assert {Path(f).name for f in mc.best_k_models} == expected

    # check created ckpts
    actual = {f.basename for f in tmpdir.listdir()}
    assert actual == {"epoch=0.ckpt", "epoch=1.ckpt", "epoch=0-v1.ckpt", "epoch=1-v1.ckpt"}


def test_ckpt_version_after_rerun_same_trainer(tmpdir):
    """Check that previous checkpoints are renamed to have the correct version suffix when the same trainer
    instance is used."""
    mc = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, monitor="epoch", filename="test")
    mc.STARTING_VERSION = 9
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        default_root_dir=tmpdir,
        callbacks=[mc],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(BoringModel())
    trainer.fit_loop.max_epochs = 4
    trainer.fit(BoringModel())

    ckpt_range = range(mc.STARTING_VERSION, trainer.max_epochs + mc.STARTING_VERSION - 1)
    expected = {"test.ckpt", *(f"test-v{i}.ckpt" for i in ckpt_range)}
    # check best_k_models state
    assert {Path(f).name for f in mc.best_k_models} == expected
    # check created ckpts
    assert set(os.listdir(tmpdir)) == expected


def test_model_checkpoint_mode_options():
    with pytest.raises(MisconfigurationException, match="`mode` can be .* but got unknown_option"):
        ModelCheckpoint(mode="unknown_option")


def test_check_val_every_n_epochs_top_k_integration(tmpdir):
    model = BoringModel()
    mc = ModelCheckpoint(dirpath=tmpdir, monitor="epoch", save_top_k=-1, filename="{epoch}")
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        max_epochs=5,
        check_val_every_n_epoch=2,
        callbacks=mc,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model)
    assert set(os.listdir(tmpdir)) == {"epoch=1.ckpt", "epoch=3.ckpt"}


def test_model_checkpoint_saveload_ckpt(tmpdir):
    def make_assertions(cb_restore, written_ckpt):
        expected_keys = {
            "dirpath": False,
            "best_model_score": False,
            "kth_best_model_path": False,
            "kth_value": False,
            "best_k_models": False,
            "best_model_path": True,
            "last_model_path": True,
        }
        for key, should_match in expected_keys.items():
            if should_match:
                assert getattr(cb_restore, key) == written_ckpt[key]
            else:
                assert getattr(cb_restore, key) != written_ckpt[key]

    class CustomModelCheckpoint(ModelCheckpoint):
        def on_load_checkpoint(self, *args, **kwargs):
            assert self.dirpath is not None
            return super().on_load_checkpoint(*args, **kwargs)

    ckpt = {
        "best_model_path": "epoch=10-step=1436.ckpt",
        "best_model_score": torch.tensor(2.246),
        "best_k_models": {"epoch=10-step=1436.ckpt": torch.tensor(2.246)},
        "kth_best_model_path": "epoch=10-step=1436.ckpt",
        "kth_value": torch.tensor(2.246),
        "last_model_path": "last2245.ckpt",
    }

    # test on_save_checkpoint
    cb_write = ModelCheckpoint(dirpath=tmpdir, save_top_k=-1, save_last=True)
    for key, val in ckpt.items():
        setattr(cb_write, key, val)
    written_ckpt = cb_write.on_save_checkpoint("", "", "")
    for state in ckpt:
        assert ckpt[state] == written_ckpt[state]

    # Case - 1
    # test on_load_checkpoint
    # Notes:
    # 1. "current_score", "dirpath" and "monitor" are currently not restored by on_load_checkpoint.
    #    We therefore set "dirpath" and "monitor" to something different than for ckpt/cb_write so we can assert them.
    # 2. "current_score" is left as initialized, i.e. None, and can therefore also be asserted
    # 3. When a different `dirpath` is passed to `ModelCheckpoint` to resume training, only
    #    `best_model_path` and `last_model_path` are reloaded (reloading for others is stopped).
    cb_restore = ModelCheckpoint(dirpath=tmpdir + "/restore", monitor=None, save_top_k=-1, save_last=True)
    with pytest.warns(UserWarning, match="The dirpath has changed from*"):
        cb_restore.on_load_checkpoint("", "", written_ckpt)
    make_assertions(cb_restore, written_ckpt)

    # Case - 2
    # Make sure that everything runs when dirpath is not initialized explicitly
    cb_restore = CustomModelCheckpoint()
    cb_restore.setup(Trainer(), BoringModel())
    with pytest.warns(UserWarning, match="The dirpath has changed from*"):
        cb_restore.on_load_checkpoint("", "", written_ckpt)
    make_assertions(cb_restore, written_ckpt)


def test_save_last_saves_correct_last_model_path(tmpdir):
    mc = ModelCheckpoint(dirpath=tmpdir, save_last=True)
    mc.CHECKPOINT_NAME_LAST = "{foo}-last"
    trainer = Trainer(callbacks=mc)
    trainer.strategy.connect(BoringModel())

    mc._save_last_checkpoint(trainer, {"foo": 1})
    expected = "foo=1-last.ckpt"
    assert os.listdir(tmpdir) == [expected]
    full_path = str(tmpdir / expected)
    ckpt = torch.load(full_path)
    assert ckpt["callbacks"][mc.state_key]["last_model_path"] == full_path


def test_none_monitor_saves_correct_best_model_path(tmpdir):
    mc = ModelCheckpoint(dirpath=tmpdir, monitor=None)
    trainer = Trainer(callbacks=mc)
    trainer.strategy.connect(BoringModel())

    mc._save_none_monitor_checkpoint(trainer, {})
    expected = "epoch=0-step=0.ckpt"
    assert os.listdir(tmpdir) == [expected]
    full_path = str(tmpdir / expected)
    ckpt = torch.load(full_path)
    assert ckpt["callbacks"][mc.state_key]["best_model_path"] == full_path


def test_last_global_step_saved():
    # this should not save anything
    model_checkpoint = ModelCheckpoint(save_top_k=0, save_last=False, monitor="foo")
    trainer = MagicMock()
    trainer.callback_metrics = {"foo": 123}
    model_checkpoint.save_checkpoint(trainer)
    assert model_checkpoint._last_global_step_saved == 0
