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
from argparse import Namespace
from copy import deepcopy
import math
import os
from pathlib import Path
import pickle
import sys
from unittest.mock import ANY, call, patch

import cloudpickle
from omegaconf import OmegaConf
import pytest
import torch

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml, save_hparams_to_tags_csv
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler.profilers import AdvancedProfiler, PassThroughProfiler, SimpleProfiler
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.utilities import NATIVE_AMP_AVAILABLE
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import BoringModel, EvalModelTemplate
import tests.base.develop_utils as tutils


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
    result = trainer.fit(model)
    # training complete
    assert result == 1, "amp + ddp model failed to complete"
    assert trainer.state == TrainerState.FINISHED

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
        if url_ckpt
        else new_weights_path
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
    result = trainer.fit(model)

    # traning complete
    assert result == 1, "amp + ddp model failed to complete"
    assert trainer.state == TrainerState.FINISHED

    # save model
    new_weights_path = os.path.join(tmpdir, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, "hparams.yaml")
    ckpt_path = (
        f"http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}"
        if url_ckpt
        else new_weights_path
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
    result = trainer.fit(model)

    # traning complete
    assert result == 1
    assert trainer.state == TrainerState.FINISHED

    # save model
    new_weights_path = os.path.join(tmpdir, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, "hparams.yaml")
    ckpt_path = (
        f"http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}"
        if url_ckpt
        else new_weights_path
    )

    try:
        EvalModelTemplate.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams_file=hparams_path,
        )
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
    except Exception:
        failed = True

    assert not failed, "Model should be loaded due to strict=False."


@pytest.mark.parametrize(
    ["schedule", "expected"],
    [pytest.param({1: 2, 3: 4}, [1, 2, 4]), pytest.param(3, [3, 3, 3]), pytest.param(4, [4, 4, 4])],
)
def test_gradient_accumulation_scheduling(tmpdir, schedule, expected):
    """
    Test grad accumulation by the freq of optimizer updates
    """

    # test incorrect configs
    with pytest.raises(IndexError):
        assert Trainer(accumulate_grad_batches={-1: 3, 1: 4, 4: 6})
    with pytest.raises(IndexError):
        assert Trainer(accumulate_grad_batches={-2: 3})

    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches={})
    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches=[[2, 3], [4, 6]])
    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches={1: 2, 3.0: 4})
    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches={1: 2.5, 3: 5})

    model = EvalModelTemplate()

    trainer = Trainer(
        accumulate_grad_batches=schedule,
        limit_train_batches=0.7,  # not to be divisible by accumulate_grad_batches on purpose
        limit_val_batches=0.8,
        max_epochs=4,
        default_root_dir=tmpdir,
    )

    model.old_optimizer_step = model.optimizer_step

    # test optimizer call freq matches scheduler
    def _optimizer_step(
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # only test the first 12 batches in epoch
        if batch_idx < 12:
            if epoch == 0:
                # reset counter when starting epoch
                if batch_idx == expected[0] - 1:
                    model.prev_called_batch_idx = expected[0] - 1

                    # use this opportunity to test once
                    assert trainer.accumulate_grad_batches == expected[0]

                # separate check for last batch with accumulate 1 step
                if expected[0] == 1 and (batch_idx + 1) == trainer.num_training_batches:
                    assert batch_idx == model.prev_called_batch_idx
                elif (batch_idx + 1) == trainer.num_training_batches:
                    # prev_called_batch_idx - schedule + modulus remainder
                    assert batch_idx == (model.prev_called_batch_idx - expected[0] + (batch_idx + 1) % expected[0])
                else:
                    assert batch_idx == model.prev_called_batch_idx
                    model.prev_called_batch_idx += expected[0]

            elif 1 <= epoch <= 2:
                # reset counter when starting epoch
                if batch_idx == expected[1] - 1:
                    model.prev_called_batch_idx = expected[1] - 1

                    # use this opportunity to test once
                    assert trainer.accumulate_grad_batches == expected[1]

                if trainer.num_training_batches == batch_idx + 1:
                    # prev_called_batch_idx - schedule + modulus remainder
                    assert batch_idx == (model.prev_called_batch_idx - expected[1] + (batch_idx + 1) % expected[1])
                else:
                    assert batch_idx == model.prev_called_batch_idx
                    model.prev_called_batch_idx += expected[1]

            else:
                if batch_idx == expected[2] - 1:
                    model.prev_called_batch_idx = expected[2] - 1

                    # use this opportunity to test once
                    assert trainer.accumulate_grad_batches == expected[2]

                if (batch_idx + 1) == trainer.num_training_batches:
                    # prev_called_batch_idx - schedule + modulus remainder
                    assert batch_idx == (model.prev_called_batch_idx - expected[2] + (batch_idx + 1) % expected[2])
                else:
                    assert batch_idx == model.prev_called_batch_idx
                    model.prev_called_batch_idx += expected[2]

        model.old_optimizer_step(
            epoch, batch_idx, optimizer, optimizer_idx, second_order_closure, on_tpu, using_native_amp, using_lbfgs
        )


@pytest.mark.parametrize(
    ["accumulate_grad_batches", "limit_train_batches"],
    [
        pytest.param({1: 2, 3: 4}, 1.0),
        pytest.param({1: 2, 3: 4}, 0.5),  # not to be divisible by accumulate_grad_batches on purpose
        pytest.param(3, 1.0),
        pytest.param(3, 0.8),  # not to be divisible by accumulate_grad_batches on purpose
        pytest.param(4, 1.0),
        pytest.param(4, 0.7),  # not to be divisible by accumulate_grad_batches on purpose
    ],
)
def test_gradient_accumulation_scheduling_last_batch(tmpdir, accumulate_grad_batches, limit_train_batches):
    """ Verify optimizer.step() applied to last batch while grad accumulation """

    class CurrentModel(BoringModel):

        def on_batch_start(self, batch, batch_idx, dataloader_idx):
            self.on_train_batch_start_state_dict = self.state_dict()

        def on_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.on_train_batch_start_end_dict = self.state_dict()
            for key in self.on_train_batch_start_end_dict.keys():
                if (batch_idx + 1) == self.trainer.num_training_batches:
                    assert torch.equal(self.on_train_batch_start_state_dict[key], self.on_train_batch_start_end_dict[key])
                else:
                    assert not torch.equal(self.on_train_batch_start_state_dict[key], self.on_train_batch_start_end_dict[key])

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


def test_dp_output_reduce():
    mixin = TrainerLoggingMixin()

    # test identity when we have a single gpu
    out = torch.rand(3, 1)
    assert mixin.reduce_distributed_output(out, num_gpus=1) is out

    # average when we have multiples
    assert mixin.reduce_distributed_output(out, num_gpus=2) == out.mean()

    # when we have a dict of vals
    out = {"a": out, "b": {"c": out}}
    reduced = mixin.reduce_distributed_output(out, num_gpus=3)
    assert reduced["a"] == out["a"]
    assert reduced["b"]["c"] == out["b"]["c"]


@pytest.mark.parametrize(
    ["save_top_k", "save_last", "file_prefix", "expected_files"],
    [
        pytest.param(
            -1,
            False,
            "",
            {"epoch=4.ckpt", "epoch=3.ckpt", "epoch=2.ckpt", "epoch=1.ckpt", "epoch=0.ckpt"},
            id="CASE K=-1  (all)",
        ),
        pytest.param(1, False, "test_prefix", {"test_prefix-epoch=4.ckpt"}, id="CASE K=1 (2.5, epoch 4)"),
        pytest.param(2, False, "", {"epoch=4.ckpt", "epoch=2.ckpt"}, id="CASE K=2 (2.5 epoch 4, 2.8 epoch 2)"),
        pytest.param(
            4,
            False,
            "",
            {"epoch=1.ckpt", "epoch=4.ckpt", "epoch=3.ckpt", "epoch=2.ckpt"},
            id="CASE K=4 (save all 4 base)",
        ),
        pytest.param(
            3, False, "", {"epoch=2.ckpt", "epoch=3.ckpt", "epoch=4.ckpt"}, id="CASE K=3 (save the 2nd, 3rd, 4th model)"
        ),
        pytest.param(1, True, "", {"epoch=4.ckpt", "last.ckpt"}, id="CASE K=1 (save the 4th model and the last model)"),
    ],
)
def test_model_checkpoint_options(tmpdir, save_top_k, save_last, file_prefix, expected_files):
    """Test ModelCheckpoint options."""

    def mock_save_function(filepath, *args):
        open(filepath, "a").close()

    # simulated losses
    losses = [10, 9, 2.8, 5, 2.5]

    checkpoint_callback = ModelCheckpoint(
        dirpath=tmpdir, filename='{epoch}', monitor='checkpoint_on', save_top_k=save_top_k,
        save_last=save_last, prefix=file_prefix, verbose=1
    )
    checkpoint_callback.save_function = mock_save_function
    trainer = Trainer()

    # emulate callback's calls during the training
    for i, loss in enumerate(losses):
        trainer.current_epoch = i
        trainer.global_step = i
        trainer.logger_connector.callback_metrics = {"checkpoint_on": torch.tensor(loss)}
        checkpoint_callback.on_validation_end(trainer, trainer.get_model())

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
    result = trainer.fit(model)
    # training complete
    assert result == 1, "training failed to complete"
    assert trainer.state == TrainerState.FINISHED

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
        num_epochs_seen = 0
        num_batches_seen = 0
        num_on_load_checkpoint_called = 0

        def on_epoch_end(self):
            self.num_epochs_seen += 1

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

    assert model.num_epochs_seen == 2
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
        new_trainer = Trainer(resume_from_checkpoint=ckpt, max_epochs=2)
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
    result = trainer.fit(model)

    assert result == 1, "Training did not complete"
    assert trainer.state == TrainerState.FINISHED
    assert trainer.global_step == trainer.max_steps, "Model did not stop at max_steps"

    # define less train epochs than steps
    trainer_kwargs['max_epochs'] = 2
    trainer_kwargs['max_steps'] = 3 * 2 * num_train_samples
    trainer = Trainer(**trainer_kwargs)
    result = trainer.fit(model)

    assert result == 1, "Training did not complete"
    assert trainer.state == TrainerState.FINISHED
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
    result = trainer.fit(model)

    assert result == 1, "Training did not complete"
    assert trainer.state == TrainerState.FINISHED
    assert trainer.current_epoch > 0
    assert trainer.global_step >= num_train_samples, "Model did not train for at least min_epochs"

    # define less epochs than min_steps
    trainer_kwargs["min_steps"] = math.floor(num_train_samples * 1.5)
    trainer = Trainer(**trainer_kwargs)
    result = trainer.fit(model)

    assert result == 1, "Training did not complete"
    assert trainer.state == TrainerState.FINISHED
    assert trainer.current_epoch > 0
    assert trainer.global_step >= math.floor(num_train_samples * 1.5), "Model did not train for at least min_steps"


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
    result = trainer.fit(model)

    assert result == 1, "Training did not complete"
    assert trainer.state == TrainerState.FINISHED
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
    result = trainer.fit(model)

    # verify training completed
    assert result == 1
    assert trainer.state == TrainerState.FINISHED

    # verify torch.backends.cudnn.benchmark is not turned off
    assert torch.backends.cudnn.benchmark


@pytest.mark.parametrize("ckpt_path", [None, "best", "specific"])
@pytest.mark.parametrize("save_top_k", [-1, 0, 1, 2])
def test_test_checkpoint_path(tmpdir, ckpt_path, save_top_k):
    hparams = EvalModelTemplate.get_default_hparams()

    model = EvalModelTemplate(**hparams)
    trainer = Trainer(
        max_epochs=2,
        progress_bar_refresh_rate=0,
        default_root_dir=tmpdir,
        callbacks=[ModelCheckpoint(monitor="early_stop_on", save_top_k=save_top_k)],
    )
    trainer.fit(model)
    if ckpt_path == "best":
        # ckpt_path is 'best', meaning we load the best weights
        if save_top_k == 0:
            with pytest.raises(MisconfigurationException, match=".*is not configured to save the best.*"):
                trainer.test(ckpt_path=ckpt_path)
        else:
            trainer.test(ckpt_path=ckpt_path)
            assert trainer.tested_ckpt_path == trainer.checkpoint_callback.best_model_path
    elif ckpt_path is None:
        # ckpt_path is None, meaning we don't load any checkpoints and
        # use the weights from the end of training
        trainer.test(ckpt_path=ckpt_path)
        assert trainer.tested_ckpt_path is None
    else:
        # specific checkpoint, pick one from saved ones
        if save_top_k == 0:
            with pytest.raises(FileNotFoundError):
                trainer.test(ckpt_path="random.ckpt")
        else:
            ckpt_path = str(
                list((Path(tmpdir) / f"lightning_logs/version_{trainer.logger.version}/checkpoints").iterdir())[
                    0
                ].absolute()
            )
            trainer.test(ckpt_path=ckpt_path)
            assert trainer.tested_ckpt_path == ckpt_path


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
    result = trainer.fit(model)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key]))

    # check that limit_train_batches=0 turns off training
    assert result == 1, "training failed to complete"
    assert trainer.state == TrainerState.FINISHED
    assert trainer.current_epoch == 0
    assert not model.training_step_invoked, "`training_step` should not run when `limit_train_batches=0`"
    assert not model.training_epoch_end_invoked, "`training_epoch_end` should not run when `limit_train_batches=0`"

    # check that limit_train_batches has no influence when fast_dev_run is turned on
    model = CurrentModel()
    trainer_options.update(fast_dev_run=True)
    before_state_dict = deepcopy(model.state_dict())

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert not torch.all(torch.eq(before_state_dict[key], after_state_dict[key]))

    assert result == 1, "training failed to complete"
    assert trainer.state == TrainerState.FINISHED
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
    result = trainer.fit(model)

    # check that limit_val_batches=0 turns off validation
    assert result == 1, "training failed to complete"
    assert trainer.state == TrainerState.FINISHED
    assert trainer.current_epoch == 1
    assert not model.validation_step_invoked, "`validation_step` should not run when `limit_val_batches=0`"
    assert not model.validation_epoch_end_invoked, "`validation_epoch_end` should not run when `limit_val_batches=0`"

    # check that limit_val_batches has no influence when fast_dev_run is turned on
    model = CurrentModel(**hparams)
    trainer_options.update(fast_dev_run=True)
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, "training failed to complete"
    assert trainer.state == TrainerState.FINISHED
    assert trainer.current_epoch == 0
    assert model.validation_step_invoked, "did not run `validation_step` with `fast_dev_run=True`"
    assert model.validation_epoch_end_invoked, "did not run `validation_epoch_end` with `fast_dev_run=True`"


def test_nan_loss_detection(tmpdir):
    class CurrentModel(EvalModelTemplate):
        test_batch_inf_loss = 8

        def training_step(self, batch, batch_idx, optimizer_idx=None):
            output = super().training_step(batch, batch_idx, optimizer_idx)
            if batch_idx == self.test_batch_inf_loss:
                if isinstance(output, dict):
                    output["loss"] *= torch.tensor(math.inf)  # make loss infinite
                else:
                    output /= 0
            return output

    model = CurrentModel()

    # fit model
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=(model.test_batch_inf_loss + 1),
        terminate_on_nan=True,
    )

    with pytest.raises(ValueError, match=r".*The loss returned in `training_step` is nan or inf.*"):
        trainer.fit(model)
        assert trainer.global_step == model.test_step_inf_loss

    for param in model.parameters():
        assert torch.isfinite(param).all()


def test_nan_params_detection(tmpdir):
    class CurrentModel(EvalModelTemplate):
        test_batch_nan = 8

        def on_after_backward(self):
            if self.global_step == self.test_batch_nan:
                # simulate parameter that became nan
                torch.nn.init.constant_(self.c_d1.bias, math.nan)

    model = CurrentModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=(model.test_batch_nan + 1),
        terminate_on_nan=True,
    )

    with pytest.raises(ValueError, match=r".*Detected nan and/or inf values in `c_d1.bias`.*"):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(not NATIVE_AMP_AVAILABLE, reason="test requires native AMP.")
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


def test_gpu_choice(tmpdir):
    trainer_options = dict(
        default_root_dir=tmpdir,
    )
    # Only run if CUDA is available
    if not torch.cuda.is_available():
        return

    num_gpus = torch.cuda.device_count()
    Trainer(**trainer_options, gpus=num_gpus, auto_select_gpus=True)

    with pytest.raises(RuntimeError, match=r".*No GPUs available.*"):
        Trainer(**trainer_options, gpus=num_gpus + 1, auto_select_gpus=True)


@pytest.mark.parametrize(
    ["limit_val_batches"],
    [
        pytest.param(0.0),  # this should run no sanity checks
        pytest.param(1),
        pytest.param(1.0),
        pytest.param(0.5),
        pytest.param(5),
    ],
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


@pytest.mark.parametrize(
    ["limit_val_batches"],
    [
        pytest.param(0.0),  # this should run no sanity checks
        pytest.param(1),
        pytest.param(1.0),
        pytest.param(0.3),
    ],
)
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
            dict(
                use_dp=False,
                use_ddp=False,
                use_ddp2=False,
                num_gpus=0,
                on_gpu=False,
                use_single_gpu=False,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator="dp", gpus=None),
            dict(
                use_dp=False,
                use_ddp=False,
                use_ddp2=False,
                num_gpus=0,
                on_gpu=False,
                use_single_gpu=False,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator="dp", gpus=None),
            dict(
                use_dp=False,
                use_ddp=False,
                use_ddp2=False,
                num_gpus=0,
                on_gpu=False,
                use_single_gpu=False,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator="ddp", gpus=None),
            dict(
                use_dp=False,
                use_ddp=False,
                use_ddp2=False,
                num_gpus=0,
                on_gpu=False,
                use_single_gpu=False,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator="ddp", num_processes=2, gpus=None),
            dict(
                use_dp=False,
                use_ddp=True,
                use_ddp2=False,
                num_gpus=0,
                on_gpu=False,
                use_single_gpu=False,
                num_processes=2,
            ),
        ),
        (
            dict(accelerator="ddp", num_nodes=2, gpus=None),
            dict(
                use_dp=False,
                use_ddp=True,
                use_ddp2=False,
                num_gpus=0,
                on_gpu=False,
                use_single_gpu=False,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator="ddp_cpu", num_processes=2, gpus=None),
            dict(
                use_dp=False,
                use_ddp=True,
                use_ddp2=False,
                num_gpus=0,
                on_gpu=False,
                use_single_gpu=False,
                num_processes=2,
            ),
        ),
        (
            dict(accelerator="ddp2", gpus=None),
            dict(
                use_dp=False,
                use_ddp=False,
                use_ddp2=False,
                num_gpus=0,
                on_gpu=False,
                use_single_gpu=False,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator=None, gpus=1),
            dict(
                use_dp=False,
                use_ddp=False,
                use_ddp2=False,
                num_gpus=1,
                on_gpu=True,
                use_single_gpu=True,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator="dp", gpus=1),
            dict(
                use_dp=True,
                use_ddp=False,
                use_ddp2=False,
                num_gpus=1,
                on_gpu=True,
                use_single_gpu=True,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator="ddp", gpus=1),
            dict(
                use_dp=False,
                use_ddp=True,
                use_ddp2=False,
                num_gpus=1,
                on_gpu=True,
                use_single_gpu=True,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator="ddp_cpu", num_processes=2, gpus=1),
            dict(
                use_dp=False,
                use_ddp=True,
                use_ddp2=False,
                num_gpus=0,
                on_gpu=False,
                use_single_gpu=False,
                num_processes=2,
            ),
        ),
        (
            dict(accelerator="ddp2", gpus=1),
            dict(
                use_dp=False,
                use_ddp=False,
                use_ddp2=True,
                num_gpus=1,
                on_gpu=True,
                use_single_gpu=False,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator=None, gpus=2),
            dict(
                use_dp=False,
                use_ddp=True,
                use_ddp2=False,
                num_gpus=2,
                on_gpu=True,
                use_single_gpu=False,
                num_processes=2,
            ),
        ),
        (
            dict(accelerator="dp", gpus=2),
            dict(
                use_dp=True,
                use_ddp=False,
                use_ddp2=False,
                num_gpus=2,
                on_gpu=True,
                use_single_gpu=False,
                num_processes=1,
            ),
        ),
        (
            dict(accelerator="ddp", gpus=2),
            dict(
                use_dp=False,
                use_ddp=True,
                use_ddp2=False,
                num_gpus=2,
                on_gpu=True,
                use_single_gpu=False,
                num_processes=2,
            ),
        ),
        (
            dict(accelerator="ddp2", gpus=2),
            dict(
                use_dp=False,
                use_ddp=False,
                use_ddp2=True,
                num_gpus=2,
                on_gpu=True,
                use_single_gpu=False,
                num_processes=1,
            ),
        ),
    ],
)
def test_trainer_config(trainer_kwargs, expected, monkeypatch):
    if trainer_kwargs["gpus"] is not None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: trainer_kwargs["gpus"])
    trainer = Trainer(**trainer_kwargs)
    assert len(expected) == 7
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
    result = trainer.fit(model)
    assert result == 1
    assert trainer.state == TrainerState.FINISHED
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
    result = trainer.fit(model)
    assert result == 1
    assert trainer.state == TrainerState.FINISHED
    assert trainer.custom_kwarg == "custom"
    assert trainer.fast_dev_run

    # when we pass in an unknown arg, the base class should complain
    with pytest.raises(TypeError, match=r"__init__\(\) got an unexpected keyword argument 'abcdefg'"):
        TrainerSubclass(abcdefg="unknown_arg")


@pytest.mark.parametrize(
    "trainer_params",
    [
        OmegaConf.create({"max_epochs": 1, "gpus": 1}),
        OmegaConf.create({"max_epochs": 1, "gpus": [0]}),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_trainer_omegaconf(trainer_params):
    Trainer(**trainer_params)


def test_trainer_pickle(tmpdir):
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
    )
    pickle.dumps(trainer)
    cloudpickle.dumps(trainer)


def test_trainer_setup_call(tmpdir):
    """Test setup call with fit and test call."""

    class CurrentModel(EvalModelTemplate):
        def setup(self, stage):
            self.stage = stage

    class TrainerSubclass(Trainer):
        def setup(self, model, stage):
            assert model is not None
            self.stage = stage

    model = CurrentModel()

    # fit model
    trainer = TrainerSubclass(default_root_dir=tmpdir, max_epochs=1, checkpoint_callback=False)

    trainer.fit(model)
    assert trainer.stage == "fit"
    assert trainer.get_model().stage == "fit"

    trainer.test(ckpt_path=None)
    assert trainer.stage == "test"
    assert trainer.get_model().stage == "test"


@pytest.mark.parametrize(
    "train_batches, max_steps, log_interval",
    [
        pytest.param(10, 10, 1),
        pytest.param(3, 10, 1),
        pytest.param(3, 10, 5),
    ],
)
@patch("pytorch_lightning.loggers.tensorboard.TensorBoardLogger.log_metrics")
def test_log_every_n_steps(log_metrics_mock, tmpdir, train_batches, max_steps, log_interval):
    model = EvalModelTemplate()
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


@pytest.mark.parametrize(['profiler', 'expected'], [
    (None, PassThroughProfiler),
    (SimpleProfiler(), SimpleProfiler),
    (AdvancedProfiler(), AdvancedProfiler),
    ('simple', SimpleProfiler),
    ('Simple', SimpleProfiler),
    ('advanced', AdvancedProfiler),
])
def test_trainer_profiler_correct_args(profiler, expected):
    kwargs = {'profiler': profiler} if profiler is not None else {}
    trainer = Trainer(**kwargs)
    assert isinstance(trainer.profiler, expected)


def test_trainer_profiler_incorrect_str_arg():
    with pytest.raises(ValueError, match=r".*can only be 'simple' or 'advanced'"):
        Trainer(profiler="unknown_profiler")


@pytest.mark.parametrize('profiler', (
    42, [42], {"a": 42}, torch.tensor(42), Trainer(),
))
def test_trainer_profiler_incorrect_arg_type(profiler):
    with pytest.raises(MisconfigurationException,
                       match=r"Only None, bool, str and subclasses of `BaseProfiler`"
                             r" are valid values for `Trainer`'s `profiler` parameter. *"):
        Trainer(profiler=profiler)
