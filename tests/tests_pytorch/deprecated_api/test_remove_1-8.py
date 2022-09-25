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
"""Test deprecated functionality which will be removed in v1.8.0."""
from unittest import mock
from unittest.mock import Mock

import pytest

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel
from pytorch_lightning.strategies.ipu import LightningIPUModule
from pytorch_lightning.trainer.configuration_validator import _check_datamodule_checkpoint_hooks
from pytorch_lightning.trainer.states import RunningStage


def test_v1_8_0_on_init_start_end(tmpdir):
    class TestCallback(Callback):
        def on_init_start(self, trainer):
            print("Starting to init trainer!")

        def on_init_end(self, trainer):
            print("Trainer is init now")

    model = BoringModel()

    trainer = Trainer(
        callbacks=[TestCallback()],
        max_epochs=1,
        fast_dev_run=True,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `on_init_start` callback hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)
    with pytest.deprecated_call(
        match="The `on_init_end` callback hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.validate(model)


def test_v1_8_0_deprecated_call_hook():
    trainer = Trainer(
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
        enable_progress_bar=False,
        logger=False,
    )
    with pytest.deprecated_call(match="was deprecated in v1.6 and will be removed in v1.8."):
        trainer.call_hook("test_hook")


@pytest.mark.parametrize("fn_prefix", ["validated", "tested", "predicted"])
def test_v1_8_0_trainer_ckpt_path_attributes(fn_prefix: str):
    test_attr = f"{fn_prefix}_ckpt_path"
    trainer = Trainer()
    with pytest.deprecated_call(match=f"{test_attr}` attribute was deprecated in v1.6 and will be removed in v1.8"):
        _ = getattr(trainer, test_attr)
    with pytest.deprecated_call(match=f"{test_attr}` attribute was deprecated in v1.6 and will be removed in v1.8"):
        setattr(trainer, test_attr, "v")


def test_v1_8_0_trainer_optimizers_mixin():
    trainer = Trainer()
    model = BoringModel()
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer

    with pytest.deprecated_call(
        match=r"`TrainerOptimizersMixin.init_optimizers` was deprecated in v1.6 and will be removed in v1.8."
    ):
        trainer.init_optimizers(model)

    with pytest.deprecated_call(
        match=r"`TrainerOptimizersMixin.convert_to_lightning_optimizers` was deprecated in v1.6 and will be removed in "
        "v1.8."
    ):
        trainer.convert_to_lightning_optimizers()


def test_v1_8_0_deprecate_trainer_data_loading_mixin():
    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    dm = BoringDataModule()
    trainer.fit(model, datamodule=dm)

    with pytest.deprecated_call(
        match=r"`TrainerDataLoadingMixin.prepare_dataloader` was deprecated in v1.6 and will be removed in v1.8.",
    ):
        trainer.prepare_dataloader(dataloader=model.train_dataloader, shuffle=False)
    with pytest.deprecated_call(
        match=r"`TrainerDataLoadingMixin.request_dataloader` was deprecated in v1.6 and will be removed in v1.8.",
    ):
        trainer.request_dataloader(stage=RunningStage.TRAINING)


def test_v1_8_0_deprecated_lightning_optimizers():
    trainer = Trainer()
    with pytest.deprecated_call(
        match="Trainer.lightning_optimizers` is deprecated in v1.6 and will be removed in v1.8"
    ):
        assert trainer.lightning_optimizers == {}


def test_v1_8_0_remove_on_batch_start_end(tmpdir):
    class TestCallback(Callback):
        def on_batch_start(self, *args, **kwargs):
            print("on_batch_start")

    model = BoringModel()
    trainer = Trainer(
        callbacks=[TestCallback()],
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `Callback.on_batch_start` hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)

    class TestCallback(Callback):
        def on_batch_end(self, *args, **kwargs):
            print("on_batch_end")

    trainer = Trainer(
        callbacks=[TestCallback()],
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `Callback.on_batch_end` hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)


def test_v1_8_0_on_configure_sharded_model(tmpdir):
    class TestCallback(Callback):
        def on_configure_sharded_model(self, trainer, model):
            print("Configuring sharded model")

    model = BoringModel()

    trainer = Trainer(
        callbacks=[TestCallback()],
        max_epochs=1,
        fast_dev_run=True,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `on_configure_sharded_model` callback hook was deprecated in v1.6 and will be removed in v1.8."
    ):
        trainer.fit(model)


def test_v1_8_0_remove_on_epoch_start_end_lightning_module(tmpdir):
    class CustomModel(BoringModel):
        def on_epoch_start(self, *args, **kwargs):
            print("on_epoch_start")

    model = CustomModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `LightningModule.on_epoch_start` hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)

    class CustomModel(BoringModel):
        def on_epoch_end(self, *args, **kwargs):
            print("on_epoch_end")

    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )

    model = CustomModel()
    with pytest.deprecated_call(
        match="The `LightningModule.on_epoch_end` hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)


def test_v1_8_0_remove_on_pretrain_routine_start_end_lightning_module(tmpdir):
    class CustomModel(BoringModel):
        def on_pretrain_routine_start(self, *args, **kwargs):
            print("foo")

    model = CustomModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `LightningModule.on_pretrain_routine_start` hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)

    class CustomModel(BoringModel):
        def on_pretrain_routine_end(self, *args, **kwargs):
            print("foo")

    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )

    model = CustomModel()
    with pytest.deprecated_call(
        match="The `LightningModule.on_pretrain_routine_end` hook was deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)


def test_v1_8_0_on_before_accelerator_backend_setup(tmpdir):
    class TestCallback(Callback):
        def on_before_accelerator_backend_setup(self, *args, **kwargs):
            print("on_before_accelerator_backend called.")

    model = BoringModel()

    trainer = Trainer(
        callbacks=[TestCallback()],
        max_epochs=1,
        fast_dev_run=True,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `on_before_accelerator_backend_setup` callback hook was deprecated in v1.6"
        " and will be removed in v1.8"
    ):
        trainer.fit(model)


def test_v1_8_0_callback_on_pretrain_routine_start_end(tmpdir):
    class TestCallback(Callback):
        def on_pretrain_routine_start(self, trainer, pl_module):
            print("on_pretrain_routine_start called.")

    model = BoringModel()

    trainer = Trainer(
        callbacks=[TestCallback()],
        fast_dev_run=True,
        enable_progress_bar=False,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `Callback.on_pretrain_routine_start` hook has been deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)

    class TestCallback(Callback):
        def on_pretrain_routine_end(self, trainer, pl_module):
            print("on_pretrain_routine_end called.")

    model = BoringModel()

    trainer = Trainer(
        callbacks=[TestCallback()],
        fast_dev_run=True,
        enable_progress_bar=False,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="The `Callback.on_pretrain_routine_end` hook has been deprecated in v1.6 and will be removed in v1.8"
    ):
        trainer.fit(model)


def test_v1_8_0_datamodule_checkpointhooks():
    class CustomBoringDataModuleSave(BoringDataModule):
        def on_save_checkpoint(self, checkpoint):
            print("override on_save_checkpoint")

    class CustomBoringDataModuleLoad(BoringDataModule):
        def on_load_checkpoint(self, checkpoint):
            print("override on_load_checkpoint")

    trainer = Mock()

    trainer.datamodule = CustomBoringDataModuleSave()
    with pytest.deprecated_call(
        match="`LightningDataModule.on_save_checkpoint` was deprecated in"
        " v1.6 and will be removed in v1.8. Use `state_dict` instead."
    ):
        _check_datamodule_checkpoint_hooks(trainer)

    trainer.datamodule = CustomBoringDataModuleLoad()
    with pytest.deprecated_call(
        match="`LightningDataModule.on_load_checkpoint` was deprecated in"
        " v1.6 and will be removed in v1.8. Use `load_state_dict` instead."
    ):
        _check_datamodule_checkpoint_hooks(trainer)


def test_v1_8_0_deprecated_lightning_ipu_module():
    with pytest.deprecated_call(match=r"has been deprecated in v1.7.0 and will be removed in v1.8."):
        _ = LightningIPUModule(BoringModel(), 32)


def test_deprecated_mc_save_checkpoint():
    mc = ModelCheckpoint()
    trainer = Trainer()
    with mock.patch.object(trainer, "save_checkpoint"), pytest.deprecated_call(
        match=r"ModelCheckpoint.save_checkpoint\(\)` was deprecated in v1.6"
    ):
        mc.save_checkpoint(trainer)


def test_v1_8_0_callback_on_load_checkpoint_hook(tmpdir):
    class TestCallbackLoadHook(Callback):
        def on_load_checkpoint(self, trainer, pl_module, callback_state):
            print("overriding on_load_checkpoint")

    model = BoringModel()
    trainer = Trainer(
        callbacks=[TestCallbackLoadHook()],
        max_epochs=1,
        fast_dev_run=True,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmpdir,
    )
    with pytest.deprecated_call(
        match="`TestCallbackLoadHook.on_load_checkpoint` will change its signature and behavior in v1.8."
        " If you wish to load the state of the callback, use `load_state_dict` instead."
        r" In v1.8 `on_load_checkpoint\(..., checkpoint\)` will receive the entire loaded"
        " checkpoint dictionary instead of callback state."
    ):
        trainer.fit(model)


def test_v1_8_0_callback_on_save_checkpoint_hook(tmpdir):
    class TestCallbackSaveHookReturn(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            return {"returning": "on_save_checkpoint"}

    class TestCallbackSaveHookOverride(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            print("overriding without returning")

    model = BoringModel()
    trainer = Trainer(
        callbacks=[TestCallbackSaveHookReturn()],
        max_epochs=1,
        fast_dev_run=True,
        enable_progress_bar=False,
        logger=False,
        default_root_dir=tmpdir,
    )
    trainer.fit(model)
    with pytest.deprecated_call(
        match="Returning a value from `TestCallbackSaveHookReturn.on_save_checkpoint` is deprecated in v1.6"
        " and will be removed in v1.8. Please override `Callback.state_dict`"
        " to return state to be saved."
    ):
        trainer.save_checkpoint(tmpdir + "/path.ckpt")

    trainer.callbacks = [TestCallbackSaveHookOverride()]
    trainer.save_checkpoint(tmpdir + "/pathok.ckpt")
