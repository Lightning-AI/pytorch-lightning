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

import pytest

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel


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
