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
import logging
import math
import os
import pickle
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from unittest.mock import ANY, call, patch

import cloudpickle
import pytest
import torch
from omegaconf import OmegaConf
from torch.optim import SGD
from torch.utils.data import DataLoader

import tests.helpers.utils as tutils
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml, save_hparams_to_tags_csv
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper, UnrepeatedDistributedSampler
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import DeviceType, DistributedType
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import seed_everything
from tests.base import EvalModelTemplate
from tests.helpers import BoringModel, RandomDataset
from tests.helpers.runif import RunIf


@pytest.mark.parametrize("url_ckpt", [True, False])
def test_no_val_module(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv("TORCH_HOME", str(tmpdir))

    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
    )
    # fit model
    trainer.fit(model)
    # training complete
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # save model
    new_weights_path = os.path.join(tmpdir, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # assert ckpt has hparams
    ckpt = torch.load(new_weights_path)
    assert LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in ckpt.keys(), "hyper_parameters missing from checkpoints"

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, "hparams.yaml")
    ckpt_path = (
        f"http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}"
        if url_ckpt else new_weights_path
    )
    model_2 = EvalModelTemplate.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams_file=hparams_path,
    )
    model_2.eval()


@pytest.mark.parametrize("url_ckpt", [True, False])
def test_no_val_end_module(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv("TORCH_HOME", tmpdir)

    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
    )
    trainer.fit(model)

    # training complete
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # save model
    new_weights_path = os.path.join(tmpdir, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, "hparams.yaml")
    ckpt_path = (
        f"http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}"
        if url_ckpt else new_weights_path
    )
    model_2 = EvalModelTemplate.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams_file=hparams_path,
    )
    model_2.eval()


@pytest.mark.parametrize("url_ckpt", [True, False])
def test_strict_model_load(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv("TORCH_HOME", tmpdir)

    model = EvalModelTemplate()
    # Extra layer
    model.c_d3 = torch.nn.Linear(model.hidden_dim, model.hidden_dim)

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmpdir)],
    )
    trainer.fit(model)

    # training complete
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # save model
    new_weights_path = os.path.join(tmpdir, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, "hparams.yaml")
    ckpt_path = (
        f"http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}"
        if url_ckpt else new_weights_path
    )

    try:
        EvalModelTemplate.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams_file=hparams_path,
        )
    # todo: specify the possible exception
    except Exception:
        failed = True
    else:
        failed = False

    assert failed, "Model should not been loaded since the extra layer added."

    failed = False
    try:
        EvalModelTemplate.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams_file=hparams_path,
            strict=False,
        )
    # todo: specify the possible exception
    except Exception:
        failed = True

    assert not failed, "Model should be loaded due to strict=False."


@pytest.mark.parametrize("accumulate_grad_batches", (1, 2, 3))
def test_trainer_accumulate_grad_batches_zero_grad(tmpdir, accumulate_grad_batches):
    with patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=20,
            limit_val_batches=1,
            max_epochs=1,
            weights_summary=None,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        trainer.fit(model)

        assert sgd_zero_grad.call_count == math.ceil(trainer.limit_train_batches / accumulate_grad_batches)


@pytest.mark.parametrize(
    ["accumulate_grad_batches", "limit_train_batches"],
    [
        ({
            1: 2,
            3: 4
        }, 1.0),
        ({
            1: 2,
            3: 4
        }, 0.5),  # not to be divisible by accumulate_grad_batches on purpose
        (3, 1.0),
        (3, 0.8),  # not to be divisible by accumulate_grad_batches on purpose
        (4, 1.0),
        (4, 0.7),  # not to be divisible by accumulate_grad_batches on purpose
    ],
)
def test_gradient_accumulation_scheduling_last_batch(tmpdir, accumulate_grad_batches, limit_train_batches):
    """ Verify optimizer.step() applied to last batch while grad accumulation """

    class CurrentModel(BoringModel):

        def on_batch_start(self, *_):
            self.on_train_batch_start_state_dict = self.state_dict()

        def on_batch_end(self, outputs, batch, batch_idx, *_):
            self.on_train_batch_start_end_dict = self.state_dict()
            for key in self.on_train_batch_start_end_dict.keys():
                equal = torch.equal(self.on_train_batch_start_state_dict[key], self.on_train_batch_start_end_dict[key])
                if (batch_idx + 1) == self.trainer.num_training_batches:
                    assert equal
                else:
                    assert not equal

    model = CurrentModel()

    trainer = Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=2,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0,
        limit_test_batches=0,
        default_root_dir=tmpdir,
    )

    trainer.fit(model)


def test_loading_meta_tags(tmpdir):
    """ test for backward compatibility to meta_tags.csv """
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()

    # save tags
    logger = tutils.get_default_logger(tmpdir)
    logger.log_hyperparams(Namespace(some_str="a_str", an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load hparams
    path_expt_dir = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(path_expt_dir, TensorBoardLogger.NAME_HPARAMS_FILE)
    hparams = load_hparams_from_yaml(hparams_path)

    # save as legacy meta_tags.csv
    tags_path = os.path.join(path_expt_dir, "meta_tags.csv")
    save_hparams_to_tags_csv(tags_path, hparams)

    tags = load_hparams_from_tags_csv(tags_path)

    assert hparams == tags


def test_loading_yaml(tmpdir):
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()

    # save tags
    logger = tutils.get_default_logger(tmpdir)
    logger.log_hyperparams(Namespace(some_str="a_str", an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load hparams
    path_expt_dir = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(path_expt_dir, "hparams.yaml")
    tags = load_hparams_from_yaml(hparams_path)

    assert tags["batch_size"] == 32 and tags["hidden_dim"] == 1000


@pytest.mark.parametrize(
    "save_top_k,save_last,expected_files",
    [
        pytest.param(-1, False, [f"epoch={i}.ckpt" for i in range(5)], id="CASE K=-1  (all)"),
        pytest.param(1, False, {"epoch=4.ckpt"}, id="CASE K=1 (2.5, epoch 4)"),
        pytest.param(2, False, [f"epoch={i}.ckpt" for i in (2, 4)], id="CASE K=2 (2.5 epoch 4, 2.8 epoch 2)"),
        pytest.param(4, False, [f"epoch={i}.ckpt" for i in range(1, 5)], id="CASE K=4 (save all 4 base)"),
        pytest.param(3, False, [f"epoch={i}.ckpt" for i in range(2, 5)], id="CASE K=3 (save the 2nd, 3rd, 4th model)"),
        pytest.param(1, True, {"epoch=4.ckpt", "last.ckpt"}, id="CASE K=1 (save the 4th model and the last model)"),
    ],
)
def test_model_checkpoint_options(tmpdir, save_top_k, save_last, expected_files):
    """Test ModelCheckpoint options."""

    def mock_save_function(filepath, *args):
        open(filepath, "a").close()

    # simulated losses
    losses = [10, 9, 2.8, 5, 2.5]

    checkpoint_callback = ModelCheckpoint(
        dirpath=tmpdir,
        filename='{epoch}',
        monitor='checkpoint_on',
        save_top_k=save_top_k,
        save_last=save_last,
        verbose=True
    )
    trainer = Trainer()
    trainer.state.fn = TrainerFn.FITTING
    trainer.save_checkpoint = mock_save_function

    # emulate callback's calls during the training
    for i, loss in enumerate(losses):
        trainer.train_loop.current_epoch = i
        trainer.train_loop.global_step = i
        trainer.logger_connector.callback_metrics = {"checkpoint_on": torch.tensor(loss)}
        checkpoint_callback.on_validation_end(trainer, trainer.lightning_module)

    file_lists = set(os.listdir(tmpdir))

    assert len(file_lists) == len(
        expected_files
    ), f"Should save {len(expected_files)} models when save_top_k={save_top_k} but found={file_lists}"

    # verify correct naming
    for fname in expected_files:
        assert fname in file_lists


def test_model_checkpoint_only_weights(tmpdir):
    """Tests use case where ModelCheckpoint is configured to save only model weights, and
    user tries to load checkpoint to resume training.
    """
    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on', save_weights_only=True)],
    )
    # fit model
    trainer.fit(model)
    # training complete
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    checkpoint_path = list(trainer.checkpoint_callback.best_k_models.keys())[0]

    # assert saved checkpoint has no trainer data
    checkpoint = torch.load(checkpoint_path)
    assert "optimizer_states" not in checkpoint, "checkpoint should contain only model weights"
    assert "lr_schedulers" not in checkpoint, "checkpoint should contain only model weights"

    # assert loading model works when checkpoint has only weights
    assert EvalModelTemplate.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # directly save model
    new_weights_path = os.path.join(tmpdir, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path, weights_only=True)
    # assert saved checkpoint has no trainer data
    checkpoint = torch.load(new_weights_path)
    assert "optimizer_states" not in checkpoint, "checkpoint should contain only model weights"
    assert "lr_schedulers" not in checkpoint, "checkpoint should contain only model weights"

    # assert restoring train state fails
    with pytest.raises(KeyError, match="checkpoint contains only the model"):
        trainer.checkpoint_connector.restore_training_state(checkpoint)


def test_model_freeze_unfreeze():
    model = EvalModelTemplate()
    model.freeze()
    model.unfreeze()


@pytest.mark.parametrize("url_ckpt", [True, False])
def test_resume_from_checkpoint_epoch_restored(monkeypatch, tmpdir, tmpdir_server, url_ckpt):
    """Verify resuming from checkpoint runs the right number of epochs"""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmpdir
    monkeypatch.setenv("TORCH_HOME", tmpdir)

    class TestModel(BoringModel):
        # Model that tracks epochs and batches seen
        num_epochs_end_seen = 0
        num_batches_seen = 0
        num_on_load_checkpoint_called = 0

        def on_epoch_end(self):
            self.num_epochs_end_seen += 1

        def on_train_batch_start(self, *_):
            self.num_batches_seen += 1

        def on_load_checkpoint(self, _):
            self.num_on_load_checkpoint_called += 1

    model = TestModel()
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=0.65,
        limit_val_batches=1,
        callbacks=[ModelCheckpoint(dirpath=tmpdir, monitor='early_stop_on', save_top_k=-1)],
        default_root_dir=tmpdir,
        val_check_interval=1.0,
        progress_bar_refresh_rate=0,
        logger=False,
        weights_summary=None,
    )
    trainer.fit(model)

    # `on_epoch_end` will be called once for val_sanity, twice for train, twice for val
    assert model.num_epochs_end_seen == 1 + 2 + 2
    assert model.num_batches_seen == trainer.num_training_batches * 2
    assert model.num_on_load_checkpoint_called == 0

    # Other checkpoints can be uncommented if/when resuming mid-epoch is supported
    checkpoints = Path(trainer.checkpoint_callback.dirpath).glob("*.ckpt")
    if url_ckpt:
        # transform local paths into url checkpoints
        ip, port = tmpdir_server
        checkpoints = [f"http://{ip}:{port}/" + ckpt.name for ckpt in checkpoints]

    for ckpt in checkpoints:
        next_model = TestModel()
        state = pl_load(ckpt)

        # Resume training
        new_trainer = Trainer(
            default_root_dir=tmpdir,
            resume_from_checkpoint=ckpt,
            max_epochs=2,
        )
        new_trainer.fit(next_model)
        assert state["global_step"] + next_model.num_batches_seen == trainer.num_training_batches * trainer.max_epochs
        assert next_model.num_on_load_checkpoint_called == 1


def test_trainer_max_steps_and_epochs(tmpdir):
    """Verify model trains according to specified max steps"""
    model = BoringModel()
    num_train_samples = math.floor(len(model.train_dataloader()) * 0.5)

    # define less train steps than epochs
    trainer_kwargs = {
        'limit_train_batches': 0.5,
        'default_root_dir': tmpdir,
        'max_epochs': 3,
        'max_steps': num_train_samples + 10,
        'logger': False,
        'weights_summary': None,
        'progress_bar_refresh_rate': 0,
    }
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_step == trainer.max_steps, "Model did not stop at max_steps"

    # define less train epochs than steps
    trainer_kwargs['max_epochs'] = 2
    trainer_kwargs['max_steps'] = 3 * 2 * num_train_samples
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_step == num_train_samples * trainer.max_epochs
    assert trainer.current_epoch == trainer.max_epochs - 1, "Model did not stop at max_epochs"


def test_trainer_min_steps_and_epochs(tmpdir):
    """Verify model trains according to specified min steps"""
    model = EvalModelTemplate()
    num_train_samples = math.floor(len(model.train_dataloader()) * 0.5)

    trainer_kwargs = {
        'limit_train_batches': 0.5,
        'default_root_dir': tmpdir,
        # define callback for stopping the model
        'callbacks': [EarlyStopping(monitor="early_stop_on", min_delta=1.0)],
        'val_check_interval': 2,
        'min_epochs': 1,
        'max_epochs': 7,
        # define less min steps than 1 epoch
        'min_steps': num_train_samples // 2,
        'logger': False,
        'weights_summary': None,
        'progress_bar_refresh_rate': 0,
    }
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch > 0
    assert trainer.global_step >= num_train_samples, "Model did not train for at least min_epochs"

    # define less epochs than min_steps
    trainer_kwargs["min_steps"] = math.floor(num_train_samples * 1.5)
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch > 0
    assert trainer.global_step >= math.floor(num_train_samples * 1.5), "Model did not train for at least min_steps"


def test_trainer_min_steps_and_min_epochs_not_reached(tmpdir, caplog):
    """ Test that min_epochs/min_steps in Trainer are enforced even if EarlyStopping is triggered. """

    class TestModel(BoringModel):
        training_step_invoked = 0

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            output["loss"] = output["loss"] * 0.0  # force minimal loss to trigger early stopping
            self.log("loss", output["loss"])
            self.training_step_invoked += 1
            assert not self.trainer.should_stop
            return output

    model = TestModel()
    early_stop = EarlyStopping(monitor="loss", patience=0, check_on_train_epoch_end=True)
    min_epochs = 5
    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        min_epochs=min_epochs,
        limit_val_batches=0,
        limit_train_batches=2,
        callbacks=[early_stop]
    )
    with caplog.at_level(logging.INFO, logger="pytorch_lightning.trainer.trainer"):
        trainer.fit(model)

    message = f"minimum epochs ({min_epochs}) or minimum steps (None) has not been met. Training will continue"
    num_messages = len([record.message for record in caplog.records if message in record.message])
    assert num_messages == min_epochs - 2
    assert model.training_step_invoked == min_epochs * 2


def test_trainer_max_steps_accumulate_batches(tmpdir):
    """Verify model trains according to specified max steps with grad accumulated batches"""
    model = BoringModel()
    num_train_samples = math.floor(len(model.train_dataloader()) * 0.5)

    # define less train steps than epochs
    trainer = Trainer(
        limit_train_batches=0.5,
        default_root_dir=tmpdir,
        max_steps=num_train_samples + 10,
        accumulate_grad_batches=10,
        logger=False,
        weights_summary=None,
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_step == trainer.max_steps, "Model did not stop at max_steps"


def test_benchmark_option(tmpdir):
    """Verify benchmark option."""

    model = EvalModelTemplate()
    model.val_dataloader = model.val_dataloader__multiple

    # verify torch.backends.cudnn.benchmark is not turned on
    assert not torch.backends.cudnn.benchmark

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        benchmark=True,
    )
    trainer.fit(model)

    # verify training completed
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # verify torch.backends.cudnn.benchmark is not turned off
    assert torch.backends.cudnn.benchmark


@pytest.mark.parametrize("ckpt_path", (None, "best", "specific"))
@pytest.mark.parametrize("save_top_k", (-1, 0, 1, 2))
@pytest.mark.parametrize("fn", ("validate", "test", "predict"))
def test_tested_checkpoint_path(tmpdir, ckpt_path, save_top_k, fn):

    class TestModel(BoringModel):

        def validation_step(self, batch, batch_idx):
            self.log("foo", -batch_idx)
            return super().validation_step(batch, batch_idx)

        def test_step(self, *args):
            return self.validation_step(*args)

        def predict_step(self, batch, *_):
            return self(batch)

    model = TestModel()
    model.test_epoch_end = None
    trainer = Trainer(
        max_epochs=2,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        progress_bar_refresh_rate=0,
        default_root_dir=tmpdir,
        callbacks=[ModelCheckpoint(monitor="foo", save_top_k=save_top_k)],
    )
    trainer.fit(model)

    trainer_fn = getattr(trainer, fn)
    path_attr = f"{fn}{'d' if fn == 'validate' else 'ed'}_ckpt_path"
    assert getattr(trainer, path_attr) is None

    if ckpt_path == "best":
        # ckpt_path is 'best', meaning we load the best weights
        if save_top_k == 0:
            with pytest.raises(MisconfigurationException, match=".*is not configured to save the best.*"):
                trainer_fn(ckpt_path=ckpt_path)
        else:
            trainer_fn(ckpt_path=ckpt_path)
            assert getattr(trainer, path_attr) == trainer.checkpoint_callback.best_model_path
    elif ckpt_path is None:
        # ckpt_path is None, meaning we don't load any checkpoints and
        # use the weights from the end of training
        trainer_fn(ckpt_path=ckpt_path)
        assert getattr(trainer, path_attr) is None
    else:
        # specific checkpoint, pick one from saved ones
        if save_top_k == 0:
            with pytest.raises(FileNotFoundError):
                trainer_fn(ckpt_path="random.ckpt")
        else:
            ckpt_path = str(
                list((Path(tmpdir) / f"lightning_logs/version_{trainer.logger.version}/checkpoints").iterdir()
                     )[0].absolute()
            )
            trainer_fn(ckpt_path=ckpt_path)
            assert getattr(trainer, path_attr) == ckpt_path


def test_disabled_training(tmpdir):
    """Verify that `limit_train_batches=0` disables the training loop unless `fast_dev_run=True`."""

    class CurrentModel(BoringModel):

        training_step_invoked = False
        training_epoch_end_invoked = False

        def training_step(self, *args, **kwargs):
            self.training_step_invoked = True
            return super().training_step(*args, **kwargs)

        def training_epoch_end(self, *args, **kwargs):
            self.training_epoch_end_invoked = True
            return super().training_epoch_end(*args, **kwargs)

    model = CurrentModel()

    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=0.0,
        limit_val_batches=0.2,
        fast_dev_run=False,
    )

    before_state_dict = deepcopy(model.state_dict())

    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key]))

    # check that limit_train_batches=0 turns off training
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 0
    assert not model.training_step_invoked, "`training_step` should not run when `limit_train_batches=0`"
    assert not model.training_epoch_end_invoked, "`training_epoch_end` should not run when `limit_train_batches=0`"

    # check that limit_train_batches has no influence when fast_dev_run is turned on
    model = CurrentModel()
    trainer_options.update(fast_dev_run=True)
    before_state_dict = deepcopy(model.state_dict())

    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert not torch.all(torch.eq(before_state_dict[key], after_state_dict[key]))

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 0
    assert model.training_step_invoked, "did not run `training_step` with `fast_dev_run=True`"
    assert model.training_epoch_end_invoked, "did not run `training_epoch_end` with `fast_dev_run=True`"


def test_disabled_validation(tmpdir):
    """Verify that `limit_val_batches=0` disables the validation loop unless `fast_dev_run=True`."""

    class CurrentModel(EvalModelTemplate):

        validation_step_invoked = False
        validation_epoch_end_invoked = False

        def validation_step(self, *args, **kwargs):
            self.validation_step_invoked = True
            return super().validation_step(*args, **kwargs)

        def validation_epoch_end(self, *args, **kwargs):
            self.validation_epoch_end_invoked = True
            return super().validation_epoch_end(*args, **kwargs)

    hparams = EvalModelTemplate.get_default_hparams()
    model = CurrentModel(**hparams)

    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=2,
        limit_train_batches=0.4,
        limit_val_batches=0.0,
        fast_dev_run=False,
    )

    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    # check that limit_val_batches=0 turns off validation
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 1
    assert not model.validation_step_invoked, "`validation_step` should not run when `limit_val_batches=0`"
    assert not model.validation_epoch_end_invoked, "`validation_epoch_end` should not run when `limit_val_batches=0`"

    # check that limit_val_batches has no influence when fast_dev_run is turned on
    model = CurrentModel(**hparams)
    trainer_options.update(fast_dev_run=True)
    trainer = Trainer(**trainer_options)
    trainer.fit(model)

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 0
    assert model.validation_step_invoked, "did not run `validation_step` with `fast_dev_run=True`"
    assert model.validation_epoch_end_invoked, "did not run `validation_epoch_end` with `fast_dev_run=True`"


def test_nan_loss_detection(tmpdir):

    class CurrentModel(BoringModel):
        test_batch_inf = 3

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            if batch_idx == self.test_batch_inf:
                if isinstance(output, dict):
                    output["loss"] *= torch.tensor(math.inf)  # make loss infinite
                else:
                    output /= 0
            return output

    model = CurrentModel()

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=(model.test_batch_inf + 1),
        terminate_on_nan=True,
    )

    with pytest.raises(ValueError, match=r".*The loss returned in `training_step` is.*"):
        trainer.fit(model)
        assert trainer.global_step == model.test_batch_inf

    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_nan_params_detection(tmpdir):

    class CurrentModel(BoringModel):
        test_batch_nan = 3

        def on_after_backward(self):
            if self.global_step == self.test_batch_nan:
                # simulate parameter that became nan
                torch.nn.init.constant_(self.layer.bias, math.nan)

    model = CurrentModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=(model.test_batch_nan + 1),
        terminate_on_nan=True,
    )

    with pytest.raises(ValueError, match=r".*Detected nan and/or inf values in `layer.bias`.*"):
        trainer.fit(model)
        assert trainer.global_step == model.test_batch_nan

    # after aborting the training loop, model still has nan-valued params
    params = torch.cat([param.view(-1) for param in model.parameters()])
    assert not torch.isfinite(params).all()


def test_trainer_interrupted_flag(tmpdir):
    """Test the flag denoting that a user interrupted training."""

    model = EvalModelTemplate()

    class InterruptCallback(Callback):

        def __init__(self):
            super().__init__()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
            raise KeyboardInterrupt

    class HandleInterruptCallback(Callback):

        def __init__(self):
            super().__init__()
            self.exc_info = None

        def on_keyboard_interrupt(self, trainer, pl_module):
            self.exc_info = sys.exc_info()

    interrupt_callback = InterruptCallback()
    handle_interrupt_callback = HandleInterruptCallback()

    trainer = Trainer(
        callbacks=[interrupt_callback, handle_interrupt_callback],
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        progress_bar_refresh_rate=0,
        logger=False,
        default_root_dir=tmpdir,
    )
    assert not trainer.interrupted
    assert handle_interrupt_callback.exc_info is None
    trainer.fit(model)
    assert trainer.interrupted
    assert isinstance(handle_interrupt_callback.exc_info[1], KeyboardInterrupt)


def test_gradient_clipping(tmpdir):
    """
    Test gradient clipping
    """
    tutils.reset_seed()

    model = EvalModelTemplate()

    trainer = Trainer(
        max_steps=1,
        max_epochs=1,
        gradient_clip_val=1.0,
        default_root_dir=tmpdir,
    )

    trainer.train_loop.old_training_step_and_backward = trainer.train_loop.training_step_and_backward

    def training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens):
        """
        wrap the forward step in a closure so second order methods work
        """
        # test that gradient is clipped correctly
        ret_val = trainer.train_loop.old_training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens)
        parameters = model.parameters()
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
        assert (grad_norm - 1.0).abs() < 0.01, "Gradient norm != 1.0: {grad_norm}".format(grad_norm=grad_norm)

        return ret_val

    trainer.train_loop.training_step_and_backward = training_step_and_backward
    # for the test
    model.prev_called_batch_idx = 0

    trainer.fit(model)


def test_gradient_clipping_by_value(tmpdir):
    """
    Test gradient clipping by value
    """
    tutils.reset_seed()

    model = BoringModel()

    grad_clip_val = 1e-10
    trainer = Trainer(
        max_steps=1,
        max_epochs=1,
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm='value',
        default_root_dir=tmpdir
    )

    trainer.train_loop.old_training_step_and_backward = trainer.train_loop.training_step_and_backward

    def training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens):
        """
        wrap the forward step in a closure so second order methods work
        """
        # test that gradient is clipped correctly
        ret_val = trainer.train_loop.old_training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens)
        parameters = model.parameters()
        grad_max_list = [torch.max(p.grad.detach().abs()) for p in parameters]
        grad_max = torch.max(torch.stack(grad_max_list))
        assert abs(grad_max.item() - grad_clip_val) < 1e-11, \
            f"Gradient max value {grad_max} != grad_clip_val {grad_clip_val} ."

        return ret_val

    trainer.train_loop.training_step_and_backward = training_step_and_backward
    # for the test
    model.prev_called_batch_idx = 0

    trainer.fit(model)


@RunIf(min_gpus=1, amp_native=True)
def test_gradient_clipping_fp16(tmpdir):
    """
    Test gradient clipping with fp16
    """
    tutils.reset_seed()

    model = EvalModelTemplate()

    trainer = Trainer(
        max_steps=1,
        max_epochs=1,
        precision=16,
        gpus=1,
        gradient_clip_val=1.0,
        default_root_dir=tmpdir,
    )

    trainer.train_loop.old_training_step_and_backward = trainer.train_loop.training_step_and_backward

    def training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens):
        """
        wrap the forward step in a closure so second order methods work
        """
        # test that gradient is clipped correctly
        ret_val = trainer.train_loop.old_training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens)
        parameters = model.parameters()
        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
        assert (grad_norm - 1.0).abs() < 0.01, "Gradient norm != 1.0: {grad_norm}".format(grad_norm=grad_norm)

        return ret_val

    trainer.train_loop.training_step_and_backward = training_step_and_backward
    model.prev_called_batch_idx = 0

    trainer.fit(model)


@RunIf(min_gpus=1, amp_native=True)
def test_gradient_clipping_by_value_fp16(tmpdir):
    """
    Test gradient clipping by value with fp16
    """
    tutils.reset_seed()

    model = BoringModel()
    grad_clip_val = 1e-10
    trainer = Trainer(
        max_steps=1,
        max_epochs=1,
        precision=16,
        gpus=1,
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm='value',
        default_root_dir=tmpdir,
    )

    trainer.train_loop.old_training_step_and_backward = trainer.train_loop.training_step_and_backward

    def training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens):
        """
        wrap the forward step in a closure so second order methods work
        """
        # test that gradient is clipped correctly
        ret_val = trainer.train_loop.old_training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens)
        parameters = model.parameters()
        grad_max_list = [torch.max(p.grad.detach().abs()) for p in parameters]
        grad_max = torch.max(torch.stack(grad_max_list))
        assert abs(grad_max.item() - grad_clip_val) < 1e-11, \
            f"Gradient max value {grad_max} != grad_clip_val {grad_clip_val} ."

        return ret_val

    trainer.train_loop.training_step_and_backward = training_step_and_backward
    model.prev_called_batch_idx = 0

    trainer.fit(model)


def test_gpu_choice(tmpdir):
    trainer_options = dict(default_root_dir=tmpdir)
    # Only run if CUDA is available
    if not torch.cuda.is_available():
        return

    num_gpus = torch.cuda.device_count()
    Trainer(**trainer_options, gpus=num_gpus, auto_select_gpus=True)

    with pytest.raises(RuntimeError, match=r".*No GPUs available.*"):
        Trainer(**trainer_options, gpus=num_gpus + 1, auto_select_gpus=True)


@pytest.mark.parametrize(
    "limit_val_batches",
    [0.0, 1, 1.0, 0.5, 5],
)
def test_num_sanity_val_steps(tmpdir, limit_val_batches):
    """
    Test that the number of sanity check batches is clipped to `limit_val_batches`.
    """
    model = EvalModelTemplate()
    model.validation_step = model.validation_step__multiple_dataloaders
    model.validation_epoch_end = model.validation_epoch_end__multiple_dataloaders
    num_sanity_val_steps = 4

    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_val_batches=limit_val_batches,
        max_steps=1,
    )
    assert trainer.num_sanity_val_steps == num_sanity_val_steps

    with patch.object(
        trainer.evaluation_loop, "evaluation_step", wraps=trainer.evaluation_loop.evaluation_step
    ) as mocked:
        val_dataloaders = model.val_dataloader__multiple_mixed_length()
        trainer.fit(model, val_dataloaders=val_dataloaders)

        assert mocked.call_count == sum(
            min(num_sanity_val_steps, num_batches) for num_batches in trainer.num_val_batches
        )


@pytest.mark.parametrize("limit_val_batches", [0.0, 1, 1.0, 0.3])
def test_num_sanity_val_steps_neg_one(tmpdir, limit_val_batches):
    """
    Test that `num_sanity_val_steps=-1` runs through all validation data once, and as many batches as
    limited by `limit_val_batches` Trainer argument.
    """
    model = EvalModelTemplate()
    model.validation_step = model.validation_step__multiple_dataloaders
    model.validation_epoch_end = model.validation_epoch_end__multiple_dataloaders
    trainer = Trainer(
        default_root_dir=tmpdir,
        num_sanity_val_steps=-1,
        limit_val_batches=limit_val_batches,
        max_steps=1,
    )
    assert trainer.num_sanity_val_steps == float("inf")

    with patch.object(
        trainer.evaluation_loop, "evaluation_step", wraps=trainer.evaluation_loop.evaluation_step
    ) as mocked:
        val_dataloaders = model.val_dataloader__multiple()
        trainer.fit(model, val_dataloaders=val_dataloaders)

        assert mocked.call_count == sum(trainer.num_val_batches)


@pytest.mark.parametrize(
    "trainer_kwargs,expected",
    [
        (
            dict(accelerator=None, gpus=None),
            dict(_distrib_type=None, _device_type=DeviceType.CPU, num_gpus=0, num_processes=1),
        ),
        (
            dict(accelerator="dp", gpus=None),
            dict(_distrib_type=None, _device_type=DeviceType.CPU, num_gpus=0, num_processes=1),
        ),
        (
            dict(accelerator="ddp", gpus=None),
            dict(_distrib_type=None, _device_type=DeviceType.CPU, num_gpus=0, num_processes=1),
        ),
        (
            dict(accelerator="ddp", num_processes=2, gpus=None),
            dict(_distrib_type=DistributedType.DDP, _device_type=DeviceType.CPU, num_gpus=0, num_processes=2),
        ),
        (
            dict(accelerator="ddp", num_nodes=2, gpus=None),
            dict(_distrib_type=DistributedType.DDP, _device_type=DeviceType.CPU, num_gpus=0, num_processes=1),
        ),
        (
            dict(accelerator="ddp_cpu", num_processes=2, gpus=None),
            dict(_distrib_type=DistributedType.DDP_SPAWN, _device_type=DeviceType.CPU, num_gpus=0, num_processes=2),
        ),
        (
            dict(accelerator="ddp2", gpus=None),
            dict(_distrib_type=None, _device_type=DeviceType.CPU, num_gpus=0, num_processes=1),
        ),
        (
            dict(accelerator=None, gpus=1),
            dict(_distrib_type=None, _device_type=DeviceType.GPU, num_gpus=1, num_processes=1),
        ),
        (
            dict(accelerator="dp", gpus=1),
            dict(_distrib_type=DistributedType.DP, _device_type=DeviceType.GPU, num_gpus=1, num_processes=1),
        ),
        (
            dict(accelerator="ddp", gpus=1),
            dict(_distrib_type=DistributedType.DDP, _device_type=DeviceType.GPU, num_gpus=1, num_processes=1),
        ),
        (
            dict(accelerator="ddp_cpu", num_processes=2, gpus=1),
            dict(_distrib_type=DistributedType.DDP_SPAWN, _device_type=DeviceType.CPU, num_gpus=0, num_processes=2),
        ),
        (
            dict(accelerator="ddp2", gpus=1),
            dict(_distrib_type=DistributedType.DDP2, _device_type=DeviceType.GPU, num_gpus=1, num_processes=1),
        ),
        (
            dict(accelerator=None, gpus=2),
            dict(_distrib_type=DistributedType.DDP_SPAWN, _device_type=DeviceType.GPU, num_gpus=2, num_processes=2),
        ),
        (
            dict(accelerator="dp", gpus=2),
            dict(_distrib_type=DistributedType.DP, _device_type=DeviceType.GPU, num_gpus=2, num_processes=1),
        ),
        (
            dict(accelerator="ddp", gpus=2),
            dict(_distrib_type=DistributedType.DDP, _device_type=DeviceType.GPU, num_gpus=2, num_processes=2),
        ),
        (
            dict(accelerator="ddp2", gpus=2),
            dict(_distrib_type=DistributedType.DDP2, _device_type=DeviceType.GPU, num_gpus=2, num_processes=1),
        ),
    ],
)
def test_trainer_config(trainer_kwargs, expected, monkeypatch):
    if trainer_kwargs["gpus"] is not None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: trainer_kwargs["gpus"])
    trainer = Trainer(**trainer_kwargs)
    assert len(expected) == 4
    for k, v in expected.items():
        assert getattr(trainer, k) == v, f"Failed {k}: {v}"


def test_trainer_subclassing():
    model = EvalModelTemplate()

    # First way of pulling out args from signature is to list them
    class TrainerSubclass(Trainer):

        def __init__(self, custom_arg, *args, custom_kwarg="test", **kwargs):
            super().__init__(*args, **kwargs)
            self.custom_arg = custom_arg
            self.custom_kwarg = custom_kwarg

    trainer = TrainerSubclass(123, custom_kwarg="custom", fast_dev_run=True)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.custom_arg == 123
    assert trainer.custom_kwarg == "custom"
    assert trainer.fast_dev_run

    # Second way is to pop from the dict
    # It's a special case because Trainer does not have any positional args
    class TrainerSubclass(Trainer):

        def __init__(self, **kwargs):
            self.custom_arg = kwargs.pop("custom_arg", 0)
            self.custom_kwarg = kwargs.pop("custom_kwarg", "test")
            super().__init__(**kwargs)

    trainer = TrainerSubclass(custom_kwarg="custom", fast_dev_run=True)
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.custom_kwarg == "custom"
    assert trainer.fast_dev_run

    # when we pass in an unknown arg, the base class should complain
    with pytest.raises(TypeError, match=r"__init__\(\) got an unexpected keyword argument 'abcdefg'"):
        TrainerSubclass(abcdefg="unknown_arg")


@pytest.mark.parametrize(
    "trainer_params", [
        OmegaConf.create(dict(max_epochs=1, gpus=1)),
        OmegaConf.create(dict(max_epochs=1, gpus=[0])),
    ]
)
@RunIf(min_gpus=1)
def test_trainer_omegaconf(trainer_params):
    Trainer(**trainer_params)


def test_trainer_pickle(tmpdir):
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
    )
    pickle.dumps(trainer)
    cloudpickle.dumps(trainer)


@pytest.mark.parametrize("stage", ("fit", "validate", "test"))
def test_trainer_setup_call(tmpdir, stage):
    """Test setup call gets the correct stage"""

    class CurrentModel(BoringModel):

        def setup(self, stage):
            self.stage = stage

    class TrainerSubclass(Trainer):

        def setup(self, model, stage):
            assert model is not None
            self.stage = stage

    model = CurrentModel()

    # fit model
    trainer = TrainerSubclass(default_root_dir=tmpdir, max_epochs=1, checkpoint_callback=False)

    if stage == "fit":
        trainer.fit(model)
    elif stage == "validate":
        trainer.validate(model, ckpt_path=None)
    else:
        trainer.test(model, ckpt_path=None)

    assert trainer.stage == stage
    assert trainer.lightning_module.stage == stage


@pytest.mark.parametrize(
    "train_batches, max_steps, log_interval",
    [
        (10, 10, 1),
        (3, 10, 1),
        (3, 10, 5),
    ],
)
@patch("pytorch_lightning.loggers.tensorboard.TensorBoardLogger.log_metrics")
def test_log_every_n_steps(log_metrics_mock, tmpdir, train_batches, max_steps, log_interval):

    class TestModel(BoringModel):

        def training_step(self, *args, **kwargs):
            self.log("foo", -1)
            return super().training_step(*args, **kwargs)

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        log_every_n_steps=log_interval,
        flush_logs_every_n_steps=log_interval,
        limit_train_batches=train_batches,
        limit_val_batches=0,
        max_steps=max_steps,
    )
    trainer.fit(model)
    expected_calls = [call(metrics=ANY, step=s) for s in range(log_interval - 1, max_steps, log_interval)]
    log_metrics_mock.assert_has_calls(expected_calls)


class TestLightningDataModule(LightningDataModule):

    def __init__(self, dataloaders):
        super().__init__()
        self._dataloaders = dataloaders

    def test_dataloader(self):
        return self._dataloaders

    def predict_dataloader(self):
        return self._dataloaders


class CustomPredictionWriter(BasePredictionWriter):

    write_on_batch_end_called = False
    write_on_epoch_end_called = False

    def __init__(self, output_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, *args, **kwargs):
        assert prediction.shape == torch.Size([1, 2])
        if trainer.accelerator_connector.is_distributed:
            assert len(batch_indices) == 1
        else:
            assert batch_indices is None
        self.write_on_batch_end_called = True

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        expected = 1 if trainer.accelerator_connector.is_distributed else 2
        assert len(predictions) == 2
        assert len(predictions[0]) == expected
        if trainer.accelerator_connector.is_distributed:
            assert len(batch_indices) == 2
            assert len(batch_indices[0]) == expected
        else:
            assert batch_indices is None
        self.write_on_epoch_end_called = True

    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        if trainer.accelerator_connector.is_distributed:
            for idx in range(2):
                assert isinstance(trainer.predict_dataloaders[idx].batch_sampler.sampler, UnrepeatedDistributedSampler)
                assert isinstance(trainer.predict_dataloaders[idx].batch_sampler, IndexBatchSamplerWrapper)
        super().on_predict_epoch_end(trainer, pl_module, outputs)


def predict(
    tmpdir, accelerator, gpus, num_processes, model=None, plugins=None, datamodule=True, pbrr=None, use_callbacks=True
):
    dataloaders = [torch.utils.data.DataLoader(RandomDataset(32, 2)), torch.utils.data.DataLoader(RandomDataset(32, 2))]

    model = model or BoringModel()
    dm = TestLightningDataModule(dataloaders)

    cb = CustomPredictionWriter(tmpdir, write_interval="batch")
    cb_1 = CustomPredictionWriter(tmpdir, write_interval="epoch")

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
        accelerator=accelerator,
        gpus=gpus,
        num_processes=num_processes,
        plugins=plugins,
        progress_bar_refresh_rate=pbrr,
        callbacks=[cb, cb_1] if use_callbacks else []
    )
    if accelerator == "ddp_spawn":
        with pytest.raises(MisconfigurationException):
            trainer.predict(model, datamodule=dm, return_predictions=True)

    if datamodule:
        results = trainer.predict(model, datamodule=dm)
    else:
        results = trainer.predict(model, dataloaders=dataloaders)

    if not isinstance(trainer.training_type_plugin, DDPSpawnPlugin):
        if use_callbacks:
            assert cb.write_on_batch_end_called
            assert not cb.write_on_epoch_end_called

            assert not cb_1.write_on_batch_end_called
            assert cb_1.write_on_epoch_end_called

        num_samples = 1 if accelerator == "ddp" else 2
        assert len(results) == 2
        assert len(results[0]) == num_samples
        assert results[0][0].shape == torch.Size([1, 2])


def test_trainer_predict_no_return(tmpdir):
    """
    Test trainer.predict warns when nothing is returned
    """

    class CustomBoringModel(BoringModel):

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            if (batch_idx + 1) % 2 == 0:
                return

            return super().predict_step(batch, batch_idx, dataloader_idx)

    with pytest.warns(UserWarning, match='predict returned None'):
        predict(tmpdir, None, None, 1, model=CustomBoringModel(), use_callbacks=False)


def test_trainer_predict_grad(tmpdir):

    class CustomBoringModel(BoringModel):

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            assert batch.expand_as(batch).grad_fn is None
            return super().predict_step(batch, batch_idx, dataloader_idx)

    predict(tmpdir, None, None, 1, model=CustomBoringModel(), use_callbacks=False)

    x = torch.zeros(1, requires_grad=True)
    assert x.expand_as(x).grad_fn is not None


@pytest.mark.parametrize('progress_bar_refresh_rate', [0, 5, None])
@pytest.mark.parametrize('datamodule', [False, True])
def test_trainer_predict_cpu(tmpdir, datamodule, progress_bar_refresh_rate):
    predict(tmpdir, None, None, 1, datamodule=datamodule, pbrr=progress_bar_refresh_rate)


@RunIf(min_gpus=2, special=True)
@pytest.mark.parametrize('num_gpus', [1, 2])
def test_trainer_predict_dp(tmpdir, num_gpus):
    predict(tmpdir, "dp", num_gpus, None)


@RunIf(min_gpus=2, special=True, fairscale=True)
def test_trainer_predict_ddp(tmpdir):
    predict(tmpdir, "ddp", 2, None)


@RunIf(min_gpus=2, skip_windows=True, special=True)
def test_trainer_predict_ddp_spawn(tmpdir):
    predict(tmpdir, "ddp_spawn", 2, None)


@RunIf(min_gpus=2, special=True)
def test_trainer_predict_1_gpu(tmpdir):
    predict(tmpdir, None, 1, None)


@RunIf(skip_windows=True)
def test_trainer_predict_ddp_cpu(tmpdir):
    predict(tmpdir, "ddp_cpu", 0, 2)


@patch('torch.cuda.device_count', return_value=2)
@patch('torch.cuda.is_available', return_value=True)
def test_spawn_predict_return_predictions(*_):
    """
    Test that `return_predictions=True` raise a MisconfigurationException with spawn training type plugins.
    """
    model = BoringModel()

    def run(expected_plugin, **trainer_kwargs):
        trainer = Trainer(**trainer_kwargs, fast_dev_run=True)
        assert isinstance(trainer.training_type_plugin, expected_plugin)
        with pytest.raises(MisconfigurationException, match="`return_predictions` should be set to `False`"):
            trainer.predict(model, dataloaders=model.train_dataloader(), return_predictions=True)

    run(DDPSpawnPlugin, accelerator="ddp_spawn", gpus=2)
    run(DDPSpawnPlugin, accelerator="ddp_cpu", num_processes=2)


@pytest.mark.parametrize("return_predictions", [None, False, True])
@pytest.mark.parametrize("precision", [32, 64])
def test_predict_return_predictions_cpu(return_predictions, precision, tmpdir):
    """
    Test that `return_predictions=True`.
    """
    seed_everything(42)
    model = BoringModel()

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, precision=precision)
    preds = trainer.predict(model, dataloaders=model.train_dataloader(), return_predictions=return_predictions)
    if return_predictions or return_predictions is None:
        assert len(preds) == 1
        assert preds[0].shape == torch.Size([1, 2])
        assert preds[0].dtype == (torch.float64 if precision == 64 else torch.float32)


@pytest.mark.parametrize(
    ["limit_train_batches", "global_step", "num_training_batches", "current_epoch", "should_train"],
    [(0.2, 0, 0, 0, False), (0.5, 10, 2, 4, True)],
)
def test_disabled_training_for_insufficient_limit_train_batches(
    tmpdir, limit_train_batches, global_step, num_training_batches, current_epoch, should_train
):
    """
    Verify when `limit_train_batches` is float & between [0.0, 1.0] and
    `int(self.num_training_batches * self.limit_train_batches) == 0`, the training loop is disabled.
    """

    class CurrentModel(BoringModel):

        training_step_invoked = False
        training_epoch_end_invoked = False

        def training_step(self, *args, **kwargs):
            self.training_step_invoked = True
            return super().training_step(*args, **kwargs)

        def training_epoch_end(self, *args, **kwargs):
            self.training_epoch_end_invoked = True
            return super().training_epoch_end(*args, **kwargs)

    dataset_len = 100
    batch_size = 25

    train = RandomDataset(32, length=dataset_len)
    train_loader = DataLoader(train, batch_size=batch_size)

    model = CurrentModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=5,
        limit_train_batches=limit_train_batches,
    )
    trainer.fit(model, train_loader)

    params_string = f"""`limit_train_batches={limit_train_batches}`, `dataset_len={dataset_len}`
                        & `batch_size={batch_size}` as
                        `num_training_batches={num_training_batches}`"""
    if should_train:
        error_string = f"should run with {params_string}"
    else:
        error_string = f"should not run with {params_string}"

    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_step == global_step
    assert trainer.num_training_batches == num_training_batches
    assert trainer.current_epoch == current_epoch
    assert model.training_step_invoked == should_train, f"`training_step` {error_string}"
    assert model.training_epoch_end_invoked == should_train, f"`training_epoch_end` {error_string}"


@pytest.mark.parametrize(["max_steps", "max_epochs", "global_step"], [(10, 5, 10), (20, None, 20)])
def test_repeated_fit_calls_with_max_epochs_and_steps(tmpdir, max_steps, max_epochs, global_step):
    """
    Ensure that the training loop is bound by `max_steps` and
    `max_epochs` for repeated calls of `trainer.fit`, and
    disabled if the limit is reached
    """

    dataset_len = 200
    batch_size = 10

    train_data = DataLoader(RandomDataset(32, dataset_len), batch_size=batch_size)

    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=max_steps,
        max_epochs=max_epochs,
    )
    trainer.fit(model, train_data)
    assert trainer.global_step == global_step
    trainer.fit(model, train_data)
    assert trainer.global_step == global_step


def test_trainer_access_in_configure_optimizers(tmpdir):
    """
    Verify that the configure optimizer function can reference the trainer.
    """

    class TestModel(BoringModel):

        def configure_optimizers(self):
            assert self.trainer is not None, "Expect to have access to the trainer within `configure_optimizers`"

    train_data = torch.utils.data.DataLoader(RandomDataset(32, 64))

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_data)


@RunIf(min_gpus=1)
def test_setup_hook_move_to_device_correctly(tmpdir):
    """
    Verify that if a user defines a layer in the setup hook function, this is moved to the correct device.
    """

    class TestModel(BoringModel):

        def setup(self, stage: str) -> None:
            self.new_layer = torch.nn.Linear(2, 2)

        def training_step(self, batch, batch_idx):
            output = self.layer(batch)
            # will crash if not moved to correct device
            output = self.new_layer(output)
            loss = self.loss(batch, output)
            return {"loss": loss}

    # fake data
    train_data = torch.utils.data.DataLoader(RandomDataset(32, 64))

    # model
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=1)
    trainer.fit(model, train_data)


def test_train_loop_system(tmpdir):
    """
    Test the following methods are called in the order in automatic optimization.
    1. optimizer.step (skip when gradient accumulation)
    2. model.training_step
    3. optimizer.zero_grad (run when the first batch of gradient accumulation)
    4. model.backward

    Note that the order is NOT `training_step`->`zero_grad`->`backward`->`step`.
    This is because `optimizer.step(closure)` calls `closure()` which then calls
    the three remaining methods `training_step`, `zero_grad` and `backward` inside.
    """
    called_methods = []

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=5,
        limit_val_batches=1,
        limit_test_batches=1,
        progress_bar_refresh_rate=0,
    )

    class TestOptimizer(SGD):

        def step(self, *args, **kwargs):
            called_methods.append("step")
            return super().step(*args, **kwargs)

        def zero_grad(self, *args, **kwargs):
            called_methods.append("zero_grad")
            return super().zero_grad(*args, **kwargs)

    class TestModel(BoringModel):

        def configure_optimizers(self):
            return TestOptimizer(self.parameters(), lr=0.1)

        def training_step(self, *args, **kwargs):
            called_methods.append("training_step")
            return super().training_step(*args, **kwargs)

        def backward(self, *args, **kwargs):
            called_methods.append("backward")
            return super().backward(*args, **kwargs)

    model = TestModel()
    trainer = Trainer(**trainer_options)

    # No methods are called yet.
    assert called_methods == []

    trainer.fit(model)
    assert called_methods == [
        "step",
        "training_step",
        "zero_grad",
        "backward",
    ] * trainer.limit_train_batches

    called_methods.clear()
    trainer = Trainer(**trainer_options, accumulate_grad_batches=3)

    # No methods are called yet.
    assert called_methods == []

    trainer.fit(model)
    assert called_methods == [
        # 0
        "training_step",
        "zero_grad",
        "backward",
        # 1
        "training_step",
        "backward",
        # 2
        "step",
        "training_step",
        "backward",
        # 3
        "training_step",
        "zero_grad",
        "backward",
        # 4
        "step",
        "training_step",
        "backward",
    ]


def test_init_optimizers_resets_lightning_optimizers(tmpdir):
    """ Test that the Trainer resets the `lightning_optimizers` list everytime new optimizers get initialized. """

    def compare_optimizers():
        assert trainer.lightning_optimizers[0].optimizer is trainer.optimizers[0]

    model = BoringModel()
    model.lr = 0.2
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        auto_lr_find=True,
    )

    trainer.tune(model)
    compare_optimizers()

    trainer.fit(model)
    compare_optimizers()

    trainer.train_loop.max_epochs = 2  # simulate multiple fit calls
    trainer.fit(model)
    compare_optimizers()


def test_check_val_every_n_epoch_exception(tmpdir):

    with pytest.raises(MisconfigurationException, match="should be an integer."):
        Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            check_val_every_n_epoch=1.2,
        )


def test_trainer_attach_data_pipeline_to_model(tmpdir):

    class DataPipeline:

        pass

    class TestDataModule(LightningDataModule):

        data_pipeline = DataPipeline()

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64))

        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64))

        def test_dataloader(self):
            return DataLoader(RandomDataset(32, 64))

    class TestCallback(Callback):

        def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
            """Called when fit begins"""
            assert isinstance(pl_module.data_pipeline, DataPipeline)

    model = BoringModel()
    dm = TestDataModule()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, callbacks=[TestCallback()])
    trainer.fit(model, datamodule=dm)


def test_exception_when_testing_or_validating_with_fast_dev_run(tmpdir):
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    model = BoringModel()
    trainer.fit(model)

    with pytest.raises(MisconfigurationException, match=r"\.validate\(\)` with `fast_dev_run=True"):
        trainer.validate()
    with pytest.raises(MisconfigurationException, match=r"\.test\(\)` with `fast_dev_run=True"):
        trainer.test()


class TrainerStagesModel(BoringModel):

    def on_train_start(self) -> None:
        assert self.trainer.model.training
        assert self.training

    def on_validation_start(self) -> None:
        assert not self.trainer.model.training
        assert not self.training

    def on_test_start(self) -> None:
        assert not self.trainer.model.training
        assert not self.training

    def on_predict_start(self) -> None:
        assert not self.trainer.model.training
        assert not self.training


@pytest.mark.parametrize(
    'accelerator,num_processes', [(None, 1), pytest.param('ddp', 2, marks=RunIf(skip_windows=True))]
)
def test_model_in_correct_mode_during_stages(tmpdir, accelerator, num_processes):
    model = TrainerStagesModel()
    trainer = Trainer(default_root_dir=tmpdir, accelerator=accelerator, num_processes=num_processes, fast_dev_run=True)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model, model.val_dataloader())


class TestDummyModelForCheckpoint(BoringModel):

    def validation_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('x', loss)

    def validation_epoch_end(self, outputs) -> None:
        pass


@RunIf(skip_windows=True)
def test_fit_test_synchronization(tmpdir):
    """Test that the trainer synchronizes processes before returning control back to the caller. """
    tutils.set_random_master_port()
    model = TestDummyModelForCheckpoint()
    checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor='x', mode='min', save_top_k=1)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        accelerator='ddp_cpu',
        num_processes=2,
        callbacks=[checkpoint],
    )
    trainer.fit(model)
    assert os.path.exists(checkpoint.best_model_path), f'Could not find checkpoint at rank {trainer.global_rank}'
    trainer.test()


class CustomCallbackOnLoadCheckpoint(Callback):

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> dict:
        return {"a": None}


def test_on_load_checkpoint_missing_callbacks(tmpdir):
    """ Test a warning appears when callbacks in the checkpoint don't match callbacks provided when resuming. """

    model = BoringModel()
    chk = ModelCheckpoint(dirpath=tmpdir, save_last=True)

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=3, callbacks=[chk, CustomCallbackOnLoadCheckpoint()])
    trainer.fit(model)

    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=5, resume_from_checkpoint=chk.last_model_path, progress_bar_refresh_rate=1
    )
    with pytest.warns(UserWarning, match="CustomCallbackOnLoadCheckpoint"):
        trainer.fit(model)


def test_module_current_fx_attributes_reset(tmpdir):
    """ Ensure that lightning module's attributes related to current fx are reset at the end of execution. """
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        checkpoint_callback=False,
        logger=False,
    )

    trainer.fit(model)
    assert model._current_fx_name is None
    assert model._current_dataloader_idx is None

    trainer.test(model)
    assert model._current_fx_name is None
    assert model._current_dataloader_idx is None


def test_exception_when_lightning_module_is_not_set_on_trainer():
    trainer = Trainer()

    with pytest.raises(MisconfigurationException, match=r"`model` must be provided.*validate"):
        trainer.validate()
    with pytest.raises(MisconfigurationException, match=r"`model` must be provided.*test"):
        trainer.test()
    with pytest.raises(MisconfigurationException, match=r"`model` must be provided.*predict"):
        trainer.predict()
