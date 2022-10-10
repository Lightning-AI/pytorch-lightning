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
import os
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.batch_size_finder import BatchSizeFinder
from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel, RandomDataset
from pytorch_lightning.utilities import AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.utils import no_warning_call


class BatchSizeDataModule(BoringDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__(data_dir)
        if batch_size is not None:
            self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.random_train, batch_size=getattr(self, "batch_size", 1))


class BatchSizeModel(BoringModel):
    def __init__(self, batch_size):
        super().__init__()
        if batch_size is not None:
            self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))

    def val_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))

    def test_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))

    def predict_dataloader(self):
        return DataLoader(RandomDataset(32, 64), batch_size=getattr(self, "batch_size", 1))


@pytest.mark.parametrize(["model_bs", "dm_bs"], [(2, -1), (2, 2), (2, None), (None, 2), (16, 16)])
def test_scale_batch_size_method_with_model_or_datamodule(tmpdir, model_bs, dm_bs):
    """Test the tuner method `Tuner.scale_batch_size` with a datamodule."""
    trainer = Trainer(default_root_dir=tmpdir, limit_train_batches=1, limit_val_batches=0, max_epochs=1)

    model = BatchSizeModel(model_bs)
    datamodule = BatchSizeDataModule(tmpdir, dm_bs) if dm_bs != -1 else None

    new_batch_size = trainer.tuner.scale_batch_size(
        model, mode="binsearch", init_val=4, max_trials=2, datamodule=datamodule
    )
    assert new_batch_size == 16

    if model_bs is not None:
        assert model.batch_size == new_batch_size
        if dm_bs == -1:
            # datamodule batch size takes precedence
            assert trainer.train_dataloader.loaders.batch_size == new_batch_size
    if dm_bs not in (-1, None):
        assert datamodule.batch_size == new_batch_size
        assert trainer.train_dataloader.loaders.batch_size == new_batch_size


@pytest.mark.parametrize("trainer_fn", ["fit", "validate", "test", "predict"])
def test_trainer_reset_correctly(tmpdir, trainer_fn):
    """Check that model and all trainer parameters are reset correctly after scaling batch size."""
    model = BatchSizeModel(batch_size=2)
    before_state_dict = deepcopy(model.state_dict())

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    changed_attributes = [
        "loggers",
        "callbacks",
        "global_step",
        "max_steps",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "limit_predict_batches",
    ]

    expected = {ca: getattr(trainer, ca) for ca in changed_attributes}
    expected_loop_state_dict = trainer.fit_loop.state_dict()

    with no_warning_call(UserWarning, match="Please add the following callbacks"):
        trainer.tuner.scale_batch_size(model, max_trials=64, method=trainer_fn)

    actual = {ca: getattr(trainer, ca) for ca in changed_attributes}
    actual_loop_state_dict = trainer.fit_loop.state_dict()
    assert expected_loop_state_dict == actual_loop_state_dict
    assert actual == expected

    after_state_dict = model.state_dict()
    for key in before_state_dict.keys():
        assert torch.all(
            torch.eq(before_state_dict[key], after_state_dict[key])
        ), "Model was not reset correctly after scaling batch size"

    assert not any(f for f in os.listdir(tmpdir) if f.startswith(".scale_batch_size_temp_model"))


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("scale_arg", ["power", "binsearch", True])
def test_auto_scale_batch_size_trainer_arg(tmpdir, scale_arg):
    """Test possible values for 'batch size auto scaling' Trainer argument."""
    before_batch_size = 2
    model = BatchSizeModel(batch_size=before_batch_size)
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, auto_scale_batch_size=scale_arg, accelerator="gpu", devices=1
    )
    trainer.tune(model)
    after_batch_size = model.batch_size
    assert before_batch_size != after_batch_size, "Batch size was not altered after running auto scaling of batch size"

    assert not any(f for f in os.listdir(tmpdir) if f.startswith(".scale_batch_size_temp_model"))


@pytest.mark.parametrize("use_hparams", [True, False])
def test_auto_scale_batch_size_set_model_attribute(tmpdir, use_hparams):
    """Test that new batch size gets written to the correct hyperparameter attribute for model."""
    hparams = {"batch_size": 2}
    before_batch_size = hparams["batch_size"]

    class HparamsBatchSizeModel(BoringModel):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.save_hyperparameters()

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=self.hparams.batch_size)

        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=self.hparams.batch_size)

    model_class = HparamsBatchSizeModel if use_hparams else BatchSizeModel
    model = model_class(**hparams)

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, auto_scale_batch_size=True)
    trainer.tune(model, scale_batch_size_kwargs={"steps_per_trial": 2, "max_trials": 4})
    after_batch_size = model.hparams.batch_size if use_hparams else model.batch_size
    assert before_batch_size != after_batch_size
    assert after_batch_size <= len(trainer.train_dataloader.dataset)


@pytest.mark.parametrize("use_hparams", [True, False])
def test_auto_scale_batch_size_set_datamodule_attribute(tmpdir, use_hparams):
    """Test that new batch size gets written to the correct hyperparameter attribute for datamodule."""
    hparams = {"batch_size": 2}
    before_batch_size = hparams["batch_size"]

    class HparamsBatchSizeDataModule(BoringDataModule):
        def __init__(self, data_dir, batch_size):
            super().__init__(data_dir)
            self.save_hyperparameters()

        def train_dataloader(self):
            return DataLoader(self.random_train, batch_size=self.hparams.batch_size)

        def val_dataloader(self):
            return DataLoader(RandomDataset(32, 64), batch_size=self.hparams.batch_size)

    datamodule_class = HparamsBatchSizeDataModule if use_hparams else BatchSizeDataModule
    datamodule = datamodule_class(data_dir=tmpdir, batch_size=before_batch_size)
    model = BatchSizeModel(**hparams)

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, auto_scale_batch_size=True)
    trainer.tune(model, datamodule=datamodule, scale_batch_size_kwargs={"steps_per_trial": 2, "max_trials": 4})
    after_batch_size = datamodule.hparams.batch_size if use_hparams else datamodule.batch_size
    assert trainer.datamodule == datamodule
    assert before_batch_size < after_batch_size
    assert after_batch_size <= len(trainer.train_dataloader.dataset)


def test_auto_scale_batch_size_duplicate_attribute_warning(tmpdir):
    """Test for a warning when model.batch_size and model.hparams.batch_size both present."""

    class TestModel(BoringModel):
        def __init__(self, batch_size=1):
            super().__init__()
            # now we have model.batch_size and model.hparams.batch_size
            self.batch_size = 1
            self.save_hyperparameters()

    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, max_epochs=1000, auto_scale_batch_size=True)
    expected_message = "Field `model.batch_size` and `model.hparams.batch_size` are mutually exclusive!"
    with pytest.warns(UserWarning, match=expected_message):
        trainer.tune(model)


@pytest.mark.parametrize("scale_method", ["power", "binsearch"])
def test_call_to_trainer_method(tmpdir, scale_method):
    """Test that calling the trainer method itself works."""
    before_batch_size = 2
    model = BatchSizeModel(batch_size=before_batch_size)

    # logger file to get meta
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1)

    after_batch_size = trainer.tuner.scale_batch_size(model, mode=scale_method, max_trials=5)
    model.batch_size = after_batch_size
    trainer.fit(model)

    assert before_batch_size != after_batch_size, "Batch size was not altered after running auto scaling of batch size"


def test_error_on_dataloader_passed_to_fit(tmpdir):
    """Verify that when the auto scale batch size feature raises an error if a train dataloader is passed to
    fit."""

    # only train passed to fit
    model = BatchSizeModel(batch_size=2)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        auto_scale_batch_size="power",
    )
    fit_options = dict(train_dataloaders=model.train_dataloader())

    with pytest.raises(
        MisconfigurationException,
        match="Batch size finder cannot be used with dataloaders passed directly",
    ):
        trainer.tune(model, **fit_options)


@RunIf(min_cuda_gpus=1)
def test_auto_scale_batch_size_with_amp(tmpdir):
    before_batch_size = 2
    model = BatchSizeModel(batch_size=before_batch_size)
    trainer = Trainer(
        default_root_dir=tmpdir, max_steps=1, auto_scale_batch_size=True, accelerator="gpu", devices=1, precision=16
    )
    trainer.tune(model)
    after_batch_size = model.batch_size
    assert trainer.amp_backend == AMPType.NATIVE
    assert trainer.scaler is not None
    assert after_batch_size != before_batch_size


def test_scale_batch_size_no_trials(tmpdir):
    """Check the result is correct even when no trials are run."""
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=1, limit_val_batches=1, limit_train_batches=1, auto_scale_batch_size="power"
    )
    model = BatchSizeModel(batch_size=2)
    result = trainer.tuner.scale_batch_size(model, max_trials=0)
    assert result == 2


def test_scale_batch_size_fails_with_unavailable_mode(tmpdir):
    """Check the tuning raises error when called with mode that does not exist."""

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.batch_size = 2

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=1,
        auto_scale_batch_size="ThisModeDoesNotExist",
    )

    with pytest.raises(ValueError, match="should be either of"):
        trainer.tune(model)
    with pytest.raises(ValueError, match="should be either of"):
        trainer.tuner.scale_batch_size(model, mode="ThisModeDoesNotExist")


@pytest.mark.parametrize("scale_method", ["power", "binsearch"])
def test_dataloader_reset_with_scale_batch_size(tmpdir, scale_method):
    """Test that train and val dataloaders are reset at every update in scale batch size."""
    model = BatchSizeModel(batch_size=16)
    max_trials = 5
    init_batch_size = 4
    scale_batch_size_kwargs = {
        "max_trials": max_trials,
        "steps_per_trial": 2,
        "init_val": init_batch_size,
        "mode": scale_method,
    }

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, auto_scale_batch_size=True)
    with patch.object(model, "on_train_epoch_end") as advance_mocked:
        new_batch_size = trainer.tune(model, scale_batch_size_kwargs=scale_batch_size_kwargs)["scale_batch_size"]
        assert advance_mocked.call_count == max_trials

    assert trainer.train_dataloader.loaders.batch_size == new_batch_size
    assert trainer.val_dataloaders[0].batch_size == init_batch_size


@pytest.mark.parametrize("trainer_fn", ["validate", "test", "predict"])
def test_tuner_with_evaluation_methods(tmpdir, trainer_fn):
    """Test batch size tuner with Trainer's evaluation methods."""
    before_batch_size = 2
    max_trials = 4
    expected_scaled_batch_size = before_batch_size ** (max_trials + 1)

    model = BatchSizeModel(batch_size=before_batch_size)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=100, auto_scale_batch_size=True)
    trainer.tune(
        model, scale_batch_size_kwargs={"max_trials": max_trials, "batch_arg_name": "batch_size"}, method=trainer_fn
    )

    after_batch_size = model.batch_size
    loop = getattr(trainer, f"{trainer_fn}_loop")

    assert trainer.global_step == 0
    assert trainer.current_epoch == 0
    assert loop.dataloader_progress.current.completed == 0
    assert loop.epoch_loop.batch_progress.current.completed == 0
    assert expected_scaled_batch_size == after_batch_size
    assert not any(f for f in os.listdir(tmpdir) if f.startswith(".scale_batch_size_temp_model"))


@pytest.mark.parametrize("trainer_fn", ["fit", "validate", "test", "predict"])
def test_batch_size_finder_callback(tmpdir, trainer_fn):
    """Test batch size finder callback with different trainer methods."""
    before_batch_size = 2
    max_trials = 4
    max_epochs = 2
    expected_scaled_batch_size = before_batch_size ** (max_trials + 1)

    model = BatchSizeModel(batch_size=before_batch_size)
    batch_size_finder = BatchSizeFinder(max_trials=max_trials, batch_arg_name="batch_size")
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=max_epochs, callbacks=[batch_size_finder])
    fn = getattr(trainer, trainer_fn)

    fn(model)
    after_batch_size = model.batch_size
    loop = getattr(trainer, f"{trainer_fn}_loop")

    if trainer_fn == "fit":
        expected_steps = trainer.train_dataloader.loaders.dataset.len // after_batch_size
        assert trainer.global_step == expected_steps * max_epochs
        assert trainer.current_epoch == max_epochs
        assert loop.epoch_loop.batch_progress.total.completed == expected_steps * max_epochs
    else:
        if trainer_fn == "validate":
            dl = trainer.val_dataloaders[0]
        elif trainer_fn == "test":
            dl = trainer.test_dataloaders[0]
        elif trainer_fn == "predict":
            dl = trainer.predict_dataloaders[0]

        expected_steps = dl.dataset.len // after_batch_size
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert loop.dataloader_progress.current.completed == 1
        assert loop.epoch_loop.batch_progress.current.completed == expected_steps

    assert expected_scaled_batch_size == after_batch_size
    assert not any(f for f in os.listdir(tmpdir) if f.startswith(".scale_batch_size_temp_model"))


def test_invalid_method_in_tuner():
    """Test that an invalid value for `method` raises an error in `Tuner`"""
    trainer = Trainer(auto_scale_batch_size=True)
    model = BoringModel()

    with pytest.raises(ValueError, match="method .* is invalid."):
        trainer.tune(model, method="prediction")


def test_if_batch_size_finder_callback_already_configured():
    """Test that an error is raised if BatchSizeFinder is already configured inside `Tuner`"""
    cb = BatchSizeFinder()
    trainer = Trainer(auto_scale_batch_size=True, callbacks=cb)
    model = BoringModel()

    with pytest.raises(MisconfigurationException, match="Trainer is already configured with a .* callback"):
        trainer.tune(model)


def test_error_if_train_or_val_dataloaders_passed_with_eval_method():
    """Test that an error is raised if `train_dataloaders` or `val_dataloaders` is passed with eval method inside
    `Tuner`"""
    trainer = Trainer(auto_scale_batch_size=True)
    model = BoringModel()
    dl = model.train_dataloader()

    with pytest.raises(MisconfigurationException, match="please consider setting `dataloaders` instead"):
        trainer.tune(model, train_dataloaders=dl, method="validate")

    with pytest.raises(MisconfigurationException, match="please consider setting `dataloaders` instead"):
        trainer.tune(model, val_dataloaders=dl, method="validate")


def test_error_if_dataloaders_passed_with_fit_method():
    """Test that an error is raised if `dataloaders` is passed with fit method inside `Tuner`"""
    trainer = Trainer(auto_scale_batch_size=True)
    model = BoringModel()
    dl = model.val_dataloader()

    with pytest.raises(
        MisconfigurationException, match="please consider setting `train_dataloaders` and `val_dataloaders` instead"
    ):
        trainer.tune(model, dataloaders=dl, method="fit")


def test_batch_size_finder_with_distributed_strategies():
    """Test that an error is raised when batch size finder is used with multi-device strategy."""
    trainer = Trainer(auto_scale_batch_size=True, devices=2, strategy="ddp", accelerator="cpu")
    model = BoringModel()
    bs_finder = BatchSizeFinder()

    with pytest.raises(
        MisconfigurationException, match="Batch size finder is not supported with distributed strategies."
    ):
        bs_finder.setup(trainer, model)


def test_batch_size_finder_with_multiple_eval_dataloaders(tmpdir):
    """Test that an error is raised with batch size finder is called with multiple eval dataloaders."""

    class CustomModel(BoringModel):
        def val_dataloader(self):
            return [super().val_dataloader(), super().val_dataloader()]

    trainer = Trainer(auto_scale_batch_size=True)
    model = CustomModel()

    with pytest.raises(
        MisconfigurationException, match="Batch size finder cannot be used with multiple .* dataloaders"
    ):
        trainer.tune(model, method="validate")


@pytest.mark.parametrize("scale_method, expected_batch_size", [("power", 62), ("binsearch", 100)])
@patch("pytorch_lightning.tuner.batch_size_scaling.is_oom_error", return_value=True)
def test_dataloader_batch_size_updated_on_failure(_, tmpdir, scale_method, expected_batch_size):
    class CustomBatchSizeModel(BatchSizeModel):
        def training_step(self, *_, **__):
            if self.batch_size > 100:
                raise RuntimeError

        def train_dataloader(self):
            return DataLoader(RandomDataset(32, 1000), batch_size=self.batch_size)

    model = CustomBatchSizeModel(batch_size=16)
    model.validation_step = None
    model.training_epoch_end = None
    scale_batch_size_kwargs = {"max_trials": 10, "steps_per_trial": 1, "init_val": 500, "mode": scale_method}

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, auto_scale_batch_size=True)
    new_batch_size = trainer.tune(model, scale_batch_size_kwargs=scale_batch_size_kwargs)["scale_batch_size"]
    assert new_batch_size == model.batch_size
    assert new_batch_size == expected_batch_size
    assert trainer.train_dataloader.loaders.batch_size == expected_batch_size
