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
import glob
import logging as log
import os
import pickle
from copy import deepcopy
from typing import Generic, Mapping, TypeVar

import cloudpickle
import pytest
import torch
from lightning.fabric import seed_everything
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.states import TrainerFn
from lightning_utilities.test.warning import no_warning_call
from torch import Tensor

import tests_pytorch.helpers.pipelines as tpipes
import tests_pytorch.helpers.utils as tutils
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel
from tests_pytorch.loops.test_loops import CustomException


class ModelTrainerPropertyParity(Callback):
    def _check_properties(self, trainer, pl_module):
        assert trainer.global_step == pl_module.global_step
        assert trainer.current_epoch == pl_module.current_epoch

    def on_train_start(self, trainer, pl_module):
        self._check_properties(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, *_):
        self._check_properties(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, *_):
        self._check_properties(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._check_properties(trainer, pl_module)


class ValTestLossBoringModel(BoringModel):
    def __init__(self, batch_size=4):
        super().__init__()
        self.save_hyperparameters()

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        self.log("val_loss", out["x"])
        return out

    def test_step(self, batch, batch_idx):
        out = super().test_step(batch, batch_idx)
        self.log("test_loss", out["y"])
        return out


T = TypeVar("T")


class GenericParentValTestLossBoringModel(Generic[T], ValTestLossBoringModel):
    def __init__(self, batch_size: int = 4):
        super().__init__(batch_size=batch_size)


class GenericValTestLossBoringModel(GenericParentValTestLossBoringModel[int]):
    pass


def test_model_properties_fit_ckpt_path(tmp_path):
    """Test that properties like `current_epoch` and `global_step` in model and trainer are always the same."""
    model = BoringModel()
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    trainer_args = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 2,
        "limit_val_batches": 2,
        "logger": False,
        "callbacks": [checkpoint_callback, ModelTrainerPropertyParity()],  # this performs the assertions
    }
    trainer = Trainer(**trainer_args)
    trainer.fit(model)

    trainer_args.update(max_epochs=2)
    trainer = Trainer(**trainer_args)
    trainer.fit(model, ckpt_path=str(tmp_path / "last.ckpt"))


@RunIf(sklearn=True)
def test_trainer_properties_restore_ckpt_path(tmp_path):
    """Test that required trainer properties are set correctly when resuming from checkpoint in different phases."""

    class CustomClassifModel(ClassificationModel):
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
            return [optimizer], [lr_scheduler]

    model = CustomClassifModel()
    dm = ClassifDataModule()
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, save_last=True)
    trainer_args = {
        "default_root_dir": tmp_path,
        "max_epochs": 1,
        "limit_train_batches": 2,
        "limit_val_batches": 2,
        "limit_test_batches": 2,
        "limit_predict_batches": 2,
        "logger": False,
        "callbacks": [checkpoint_callback],
        "num_sanity_val_steps": 0,
    }
    trainer = Trainer(**trainer_args)
    trainer.fit(model, datamodule=dm)

    resume_ckpt = str(tmp_path / "last.ckpt")
    state_dict = torch.load(resume_ckpt)

    trainer_args.update({"max_epochs": 3, "enable_checkpointing": False, "callbacks": []})

    class CustomClassifModel(CustomClassifModel):
        def _is_equal(self, a, b):
            if isinstance(a, Tensor):
                return torch.all(torch.eq(a, b))

            if isinstance(a, Mapping):
                return all(self._is_equal(a.get(k, None), b.get(k, None)) for k in b)

            return a == b

        def _check_optimizers(self):
            return all(
                self._is_equal(optimizer.state_dict(), state)
                for optimizer, state in zip(self.trainer.optimizers, state_dict["optimizer_states"])
            )

        def _check_schedulers(self):
            return all(
                self._is_equal(config.scheduler.state_dict(), state)
                for config, state in zip(self.trainer.lr_scheduler_configs, state_dict["lr_schedulers"])
            )

        def _check_model_state_dict(self):
            return all(
                self._is_equal(actual, expected)
                for actual, expected in zip(self.state_dict(), state_dict["state_dict"])
            )

        def _test_on_val_test_predict_start(self):
            assert self.trainer.current_epoch == state_dict["epoch"]
            assert self.trainer.global_step == 0
            assert self._check_model_state_dict()

        def on_train_start(self):
            assert self.trainer.current_epoch == state_dict["epoch"] + 1
            assert self.trainer.global_step == state_dict["global_step"]
            assert self._check_model_state_dict()
            assert self._check_optimizers()
            assert self._check_schedulers()

        def on_validation_start(self):
            if self.trainer.state.fn == TrainerFn.VALIDATING:
                self._test_on_val_test_predict_start()

        def on_test_start(self):
            self._test_on_val_test_predict_start()

    for fn in ("fit", "validate", "test", "predict"):
        model = CustomClassifModel()
        dm = ClassifDataModule()
        trainer = Trainer(**trainer_args)
        trainer_fn = getattr(trainer, fn)
        trainer_fn(model, datamodule=dm, ckpt_path=resume_ckpt)


def test_correct_step_and_epoch(tmp_path):
    model = BoringModel()
    first_max_epochs = 2
    train_batches = 2
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=first_max_epochs, limit_train_batches=train_batches, limit_val_batches=0
    )
    assert trainer.current_epoch == 0
    assert trainer.global_step == 0

    trainer.fit(model)
    assert trainer.current_epoch == first_max_epochs
    assert trainer.global_step == first_max_epochs * train_batches

    # save checkpoint after loop ends, training end called, epoch count increased
    ckpt_path = str(tmp_path / "model.ckpt")
    trainer.save_checkpoint(ckpt_path)

    ckpt = torch.load(ckpt_path)
    assert ckpt["epoch"] == first_max_epochs
    assert ckpt["global_step"] == first_max_epochs * train_batches

    max_epochs = first_max_epochs + 2
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=max_epochs, limit_train_batches=train_batches, limit_val_batches=0
    )
    # the ckpt state is not loaded at this point
    assert trainer.current_epoch == 0
    assert trainer.global_step == 0

    class TestModel(BoringModel):
        def on_train_start(self) -> None:
            assert self.trainer.current_epoch == first_max_epochs
            assert self.trainer.global_step == first_max_epochs * train_batches
            assert self.trainer.fit_loop.epoch_loop._batches_that_stepped == first_max_epochs * train_batches

    trainer.fit(TestModel(), ckpt_path=ckpt_path)
    assert trainer.current_epoch == max_epochs
    assert trainer.global_step == max_epochs * train_batches
    assert trainer.fit_loop.epoch_loop._batches_that_stepped == max_epochs * train_batches


def test_fit_twice(tmp_path):
    epochs = []

    class TestModel(BoringModel):
        def on_train_epoch_end(self, *_):
            epochs.append(self.current_epoch)

    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(TestModel())
    trainer.fit_loop.max_epochs = 4
    trainer.fit(TestModel())
    assert epochs == [0, 1, 2, 3]


def test_try_resume_from_non_existing_checkpoint(tmp_path):
    """Test that trying to resume from non-existing `ckpt_path` fails with an error."""
    model = BoringModel()
    trainer = Trainer(logger=False)

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        trainer.fit(model, ckpt_path=str(tmp_path / "non_existing.ckpt"))


class CaptureCallbacksBeforeTraining(Callback):
    callbacks = []

    def on_fit_start(self, trainer, pl_module):
        self.callbacks = deepcopy(trainer.callbacks)


@RunIf(sklearn=True)
def test_callbacks_state_fit_ckpt_path(tmp_path):
    """Test that resuming from a checkpoint restores callbacks that persist state."""
    dm = ClassifDataModule()
    model = ClassificationModel()
    callback_capture = CaptureCallbacksBeforeTraining()

    def get_trainer_args():
        checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", save_last=True)
        assert checkpoint.best_model_path == ""
        assert checkpoint.best_model_score is None
        return {
            "default_root_dir": tmp_path,
            "limit_train_batches": 1,
            "limit_val_batches": 2,
            "max_epochs": 1,
            "logger": False,
            "callbacks": [checkpoint, callback_capture],
        }

    # initial training
    trainer = Trainer(**get_trainer_args())
    trainer.fit(model, datamodule=dm)

    callbacks_before_resume = deepcopy(trainer.callbacks)

    # resumed training
    trainer = Trainer(**get_trainer_args())
    trainer.fit(model, datamodule=dm, ckpt_path=str(tmp_path / "last.ckpt"))

    assert len(callbacks_before_resume) == len(callback_capture.callbacks)

    for before, after in zip(callbacks_before_resume, callback_capture.callbacks):
        if isinstance(before, ModelCheckpoint):
            for attribute in (
                "best_model_path",
                "best_model_score",
                "best_k_models",
                "kth_best_model_path",
                "kth_value",
                "last_model_path",
            ):
                assert getattr(before, attribute) == getattr(after, attribute), f"{attribute}"


@RunIf(sklearn=True)
def test_callbacks_references_fit_ckpt_path(tmp_path):
    """Test that resuming from a checkpoint sets references as expected."""
    dm = ClassifDataModule()
    model = ClassificationModel()
    args = {
        "default_root_dir": tmp_path,
        "max_steps": 1,
        "logger": False,
        "limit_val_batches": 2,
        "num_sanity_val_steps": 0,
    }

    # initial training
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", save_last=True)
    trainer = Trainer(**args, callbacks=[checkpoint])
    assert checkpoint is trainer.callbacks[-1] is trainer.checkpoint_callback
    trainer.fit(model, datamodule=dm)

    # resumed training
    new_checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", save_last=True)
    # pass in a new checkpoint object, which should take
    # precedence over the one in the last.ckpt file
    trainer = Trainer(**args, callbacks=[new_checkpoint])
    assert checkpoint is not new_checkpoint
    assert new_checkpoint is trainer.callbacks[-1] is trainer.checkpoint_callback
    trainer.fit(model, datamodule=dm, ckpt_path=str(tmp_path / "last.ckpt"))


@RunIf(min_cuda_gpus=2, sklearn=True)
def test_running_test_pretrained_model_distrib_ddp_spawn(tmp_path):
    """Verify `test()` on pretrained model."""
    dm = ClassifDataModule()
    model = ClassificationModel()

    # exp file to get meta
    logger = tutils.get_default_logger(tmp_path)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = {
        "enable_progress_bar": False,
        "max_epochs": 2,
        "limit_train_batches": 2,
        "limit_val_batches": 2,
        "callbacks": [checkpoint],
        "logger": logger,
        "accelerator": "gpu",
        "devices": [0, 1],
        "strategy": "ddp_spawn",
        "default_root_dir": tmp_path,
    }

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model, datamodule=dm)

    log.info(os.listdir(tutils.get_data_path(logger, path_dir=tmp_path)))

    # correct result and ok accuracy
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    pretrained_model = ClassificationModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # run test set
    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model, datamodule=dm)
    pretrained_model.cpu()

    dataloaders = dm.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        tpipes.run_model_prediction(pretrained_model, dataloader, min_acc=0.1)


@RunIf(sklearn=True)
def test_running_test_pretrained_model_cpu(tmp_path):
    """Verify test() on pretrained model."""
    seed_everything(1)

    dm = ClassifDataModule()
    model = ClassificationModel()

    # logger file to get meta
    logger = tutils.get_default_logger(tmp_path)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = {
        "enable_progress_bar": False,
        "max_epochs": 2,
        "limit_train_batches": 2,
        "limit_val_batches": 2,
        "limit_test_batches": 2,
        "callbacks": [checkpoint],
        "logger": logger,
        "default_root_dir": tmp_path,
    }

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model, datamodule=dm)

    # correct result and ok accuracy
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    pretrained_model = ClassificationModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model, datamodule=dm)

    # test we have good test accuracy
    tutils.assert_ok_model_acc(new_trainer, key="test_acc", thr=0.45)


@pytest.mark.parametrize("model_template", [ValTestLossBoringModel, GenericValTestLossBoringModel])
def test_load_model_from_checkpoint(tmp_path, model_template):
    """Verify test() on pretrained model."""
    model = model_template()

    trainer_options = {
        "enable_progress_bar": False,
        "max_epochs": 2,
        "limit_train_batches": 2,
        "limit_val_batches": 2,
        "limit_test_batches": 2,
        "callbacks": [ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", save_top_k=-1)],
        "default_root_dir": tmp_path,
        "accelerator": "cpu",
    }

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.fit(model)
    trainer.test(model)

    # correct result and ok accuracy
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # load last checkpoint
    last_checkpoint = sorted(glob.glob(os.path.join(trainer.checkpoint_callback.dirpath, "*.ckpt")))[-1]

    # Since `BoringModel` has `_save_hparams = True` by default, check that ckpt has hparams
    ckpt = torch.load(last_checkpoint)
    assert model_template.CHECKPOINT_HYPER_PARAMS_KEY in ckpt, "hyper_parameters missing from checkpoints"

    # Ensure that model can be correctly restored from checkpoint
    pretrained_model = model_template.load_from_checkpoint(last_checkpoint)

    # test that hparams loaded correctly
    for k, v in model.hparams.items():
        assert getattr(pretrained_model.hparams, k) == v

    # assert weights are the same
    for (old_name, old_p), (new_name, new_p) in zip(model.named_parameters(), pretrained_model.named_parameters()):
        assert torch.all(torch.eq(old_p, new_p)), "loaded weights are not the same as the saved weights"

    # Check `test` on pretrained model:
    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)


def test_model_saving_loading(tmp_path):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    model = BoringModel()

    # logger file to get meta
    logger = tutils.get_default_logger(tmp_path)

    # fit model
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmp_path)],
        default_root_dir=tmp_path,
    )
    trainer.fit(model)

    # make a prediction
    dataloaders = model.test_dataloader()
    batch = next(iter(dataloaders))

    # generate preds before saving model
    model.eval()
    pred_before_saving = model(batch)

    # save model
    new_weights_path = os.path.join(tmp_path, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmp_path)
    hparams_path = os.path.join(hparams_path, "hparams.yaml")
    model_2 = BoringModel.load_from_checkpoint(checkpoint_path=new_weights_path, hparams_file=hparams_path)
    model_2.eval()

    # make prediction
    # assert that both predictions are the same
    new_pred = model_2(batch)
    assert torch.all(torch.eq(pred_before_saving, new_pred)).item() == 1


@pytest.mark.parametrize("url_ckpt", [True, False])
def test_strict_model_load_more_params(monkeypatch, tmp_path, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmp_path
    monkeypatch.setenv("TORCH_HOME", tmp_path)

    model = BoringModel()
    # Extra layer
    model.c_d3 = torch.nn.Linear(32, 32)

    # logger file to get meta
    logger = tutils.get_default_logger(tmp_path)

    # fit model
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmp_path)],
    )
    trainer.fit(model)

    # traning complete
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # save model
    new_weights_path = os.path.join(tmp_path, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = os.path.join(tutils.get_data_path(logger, path_dir=tmp_path), "hparams.yaml")
    hparams_url = f"http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}"
    ckpt_path = hparams_url if url_ckpt else new_weights_path

    BoringModel.load_from_checkpoint(checkpoint_path=ckpt_path, hparams_file=hparams_path, strict=False)

    with pytest.raises(RuntimeError, match=r'Unexpected key\(s\) in state_dict: "c_d3.weight", "c_d3.bias"'):
        BoringModel.load_from_checkpoint(checkpoint_path=ckpt_path, hparams_file=hparams_path, strict=True)


@pytest.mark.parametrize("url_ckpt", [True, False])
def test_strict_model_load_less_params(monkeypatch, tmp_path, tmpdir_server, url_ckpt):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    # set $TORCH_HOME, which determines torch hub's cache path, to tmp_path
    monkeypatch.setenv("TORCH_HOME", tmp_path)

    model = BoringModel()

    # logger file to get meta
    logger = tutils.get_default_logger(tmp_path)

    # fit model
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        logger=logger,
        callbacks=[ModelCheckpoint(dirpath=tmp_path)],
    )
    trainer.fit(model)

    # traning complete
    assert trainer.state.finished, f"Training failed with {trainer.state}"

    # save model
    new_weights_path = os.path.join(tmp_path, "save_test.ckpt")
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = os.path.join(tutils.get_data_path(logger, path_dir=tmp_path), "hparams.yaml")
    ckpt_url = f"http://{tmpdir_server[0]}:{tmpdir_server[1]}/{os.path.basename(new_weights_path)}"
    ckpt_path = ckpt_url if url_ckpt else new_weights_path

    class CurrentModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.c_d3 = torch.nn.Linear(7, 7)

    CurrentModel.load_from_checkpoint(checkpoint_path=ckpt_path, hparams_file=hparams_path, strict=False)

    with pytest.raises(RuntimeError, match=r'Missing key\(s\) in state_dict: "c_d3.weight", "c_d3.bias"'):
        CurrentModel.load_from_checkpoint(checkpoint_path=ckpt_path, hparams_file=hparams_path, strict=True)


def test_model_pickle(tmp_path):
    model = BoringModel()
    pickle.dumps(model)
    cloudpickle.dumps(model)


class ExceptionModel(BoringModel):
    def __init__(self, stop_batch_idx):
        super().__init__()
        self.stop_batch_idx = stop_batch_idx

    def training_step(self, batch, batch_idx):
        if batch_idx == self.stop_batch_idx:
            raise CustomException()
        return super().training_step(batch, batch_idx)


class ShouldStopModel(ExceptionModel):
    def training_step(self, batch, batch_idx):
        if batch_idx == self.stop_batch_idx:
            # setting should_stop is treated differently to raising an exception.
            # checking both tests that this warning is raised in the correct loop
            self.trainer.should_stop = True
        return super().training_step(batch, batch_idx)


@pytest.mark.parametrize("stop_in_the_middle", [True, False])
@pytest.mark.parametrize("model_cls", [ExceptionModel, ShouldStopModel])
def test_restarting_mid_epoch_raises_warning(tmp_path, stop_in_the_middle, model_cls):
    """Test that a warning is raised if training is restarted from mid-epoch."""
    limit_train_batches = 8
    trainer_kwargs = {
        "default_root_dir": tmp_path,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": 0,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    trainer = Trainer(max_epochs=1, **trainer_kwargs)
    model = model_cls(limit_train_batches // 2 if stop_in_the_middle else -1)

    if stop_in_the_middle:
        with pytest.raises(CustomException):
            trainer.fit(model)
    else:
        trainer.fit(model)

    ckpt_path = str(tmp_path / "resume.ckpt")
    trainer.save_checkpoint(ckpt_path)

    trainer = Trainer(max_epochs=2, **trainer_kwargs)
    model.stop_batch_idx = -1

    context_manager = pytest.warns if stop_in_the_middle else no_warning_call
    with context_manager(UserWarning, match="resuming from a checkpoint that ended"):
        trainer.fit(model, ckpt_path=ckpt_path)

    if stop_in_the_middle:
        trainer = Trainer(max_epochs=2, **trainer_kwargs)
        with pytest.warns(UserWarning, match="resuming from a checkpoint that ended"):
            trainer.fit(model, ckpt_path=ckpt_path)
