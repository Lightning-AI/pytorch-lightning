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
"""Test deprecated functionality which will be removed in v2.0.0."""
from unittest import mock

import pytest

import pytorch_lightning
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel
from tests_pytorch.callbacks.test_callbacks import OldStatefulCallback
from tests_pytorch.helpers.runif import RunIf


def test_v2_0_0_deprecated_num_processes():
    with pytest.deprecated_call(match=r"is deprecated in v1.7 and will be removed in v2.0."):
        _ = Trainer(num_processes=2)


def test_v2_0_0_deprecated_gpus(cuda_count_4):
    with pytest.deprecated_call(match=r"is deprecated in v1.7 and will be removed in v2.0."):
        _ = Trainer(gpus=0)


@RunIf(skip_windows=True)
def test_v2_0_0_deprecated_tpu_cores(tpu_available):
    with pytest.deprecated_call(match=r"is deprecated in v1.7 and will be removed in v2.0."):
        _ = Trainer(tpu_cores=8)


@mock.patch("pytorch_lightning.accelerators.ipu.IPUAccelerator.is_available", return_value=True)
def test_v2_0_0_deprecated_ipus(_, monkeypatch):
    monkeypatch.setattr(pytorch_lightning.strategies.ipu, "_IPU_AVAILABLE", True)
    with pytest.deprecated_call(match=r"is deprecated in v1.7 and will be removed in v2.0."):
        _ = Trainer(ipus=4)


def test_v2_0_0_resume_from_checkpoint_trainer_constructor(tmpdir):
    # test resume_from_checkpoint still works until v2.0 deprecation
    model = BoringModel()
    callback = OldStatefulCallback(state=111)
    trainer = Trainer(default_root_dir=tmpdir, max_steps=1, callbacks=[callback])
    trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path

    callback = OldStatefulCallback(state=222)
    with pytest.deprecated_call(match=r"Setting `Trainer\(resume_from_checkpoint=\)` is deprecated in v1.5"):
        trainer = Trainer(default_root_dir=tmpdir, max_steps=2, callbacks=[callback], resume_from_checkpoint=ckpt_path)
    with pytest.deprecated_call(match=r"trainer.resume_from_checkpoint` is deprecated in v1.5"):
        _ = trainer.resume_from_checkpoint
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path == ckpt_path
    trainer.validate(model=model, ckpt_path=ckpt_path)
    assert callback.state == 222
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path == ckpt_path
    with pytest.deprecated_call(match=r"trainer.resume_from_checkpoint` is deprecated in v1.5"):
        trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path  # last `fit` replaced the `best_model_path`
    assert callback.state == 111
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path is None
    trainer.predict(model=model, ckpt_path=ckpt_path)
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path is None
    trainer.fit(model)
    assert trainer._checkpoint_connector.resume_checkpoint_path is None
    assert trainer._checkpoint_connector.resume_from_checkpoint_fit_path is None

    # test fit(ckpt_path=) precedence over Trainer(resume_from_checkpoint=) path
    model = BoringModel()
    with pytest.deprecated_call(match=r"Setting `Trainer\(resume_from_checkpoint=\)` is deprecated in v1.5"):
        trainer = Trainer(resume_from_checkpoint="trainer_arg_path")
    with pytest.raises(FileNotFoundError, match="Checkpoint at fit_arg_ckpt_path not found. Aborting training."):
        trainer.fit(model, ckpt_path="fit_arg_ckpt_path")


def test_v2_0_0_callback_on_load_checkpoint_hook(tmpdir):
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
    with pytest.raises(
        RuntimeError, match="`TestCallbackLoadHook.on_load_checkpoint` has changed its signature and behavior in v1.8."
    ):
        trainer.fit(model)


def test_v2_0_0_callback_on_save_checkpoint_hook(tmpdir):
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
    with pytest.raises(
        ValueError,
        match=(
            "Returning a value from `TestCallbackSaveHookReturn.on_save_checkpoint` was deprecated in v1.6 and is"
            " no longer supported as of v1.8"
        ),
    ):
        trainer.save_checkpoint(tmpdir + "/path.ckpt")

    trainer.callbacks = [TestCallbackSaveHookOverride()]
    trainer.save_checkpoint(tmpdir + "/pathok.ckpt")


def test_v2_0_0_remove_on_batch_start_end(tmpdir):
    class TestCallback(Callback):
        def on_batch_start(self, *args, **kwargs):
            print("on_batch_start")

    model = BoringModel()
    trainer = Trainer(
        callbacks=[TestCallback()],
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )
    with pytest.raises(RuntimeError, match="The `Callback.on_batch_start` hook was removed in v1.8"):
        trainer.fit(model)

    class TestCallback(Callback):
        def on_batch_end(self, *args, **kwargs):
            print("on_batch_end")

    trainer = Trainer(
        callbacks=[TestCallback()],
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )
    with pytest.raises(RuntimeError, match="The `Callback.on_batch_end` hook was removed in v1.8"):
        trainer.fit(model)


def test_v2_0_0_on_configure_sharded_model(tmpdir):
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
    with pytest.raises(RuntimeError, match="The `on_configure_sharded_model` callback hook was removed in v1.8."):
        trainer.fit(model)


def test_v2_0_0_remove_on_epoch_start_end_lightning_module(tmpdir):
    class CustomModel(BoringModel):
        def on_epoch_start(self, *args, **kwargs):
            print("on_epoch_start")

    model = CustomModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )
    with pytest.raises(RuntimeError, match="The `LightningModule.on_epoch_start` hook was removed in v1.8"):
        trainer.fit(model)

    class CustomModel(BoringModel):
        def on_epoch_end(self, *args, **kwargs):
            print("on_epoch_end")

    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )

    model = CustomModel()
    with pytest.raises(RuntimeError, match="The `LightningModule.on_epoch_end` hook was removed in v1.8"):
        trainer.fit(model)


def test_v2_0_0_remove_on_pretrain_routine_start_end_lightning_module(tmpdir):
    class CustomModel(BoringModel):
        def on_pretrain_routine_start(self, *args, **kwargs):
            print("foo")

    model = CustomModel()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )
    with pytest.raises(RuntimeError, match="The `LightningModule.on_pretrain_routine_start` hook was removed in v1.8"):
        trainer.fit(model)

    class CustomModel(BoringModel):
        def on_pretrain_routine_end(self, *args, **kwargs):
            print("foo")

    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
    )

    model = CustomModel()
    with pytest.raises(RuntimeError, match="The `LightningModule.on_pretrain_routine_end` hook was removed in v1.8"):
        trainer.fit(model)


def test_v2_0_0_on_before_accelerator_backend_setup(tmpdir):
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
    with pytest.raises(
        RuntimeError, match="The `on_before_accelerator_backend_setup` callback hook was removed in v1.8"
    ):
        trainer.fit(model)


def test_v2_0_0_callback_on_pretrain_routine_start_end(tmpdir):
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
    with pytest.raises(RuntimeError, match="The `Callback.on_pretrain_routine_start` hook was removed in v1.8"):
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
    with pytest.raises(RuntimeError, match="The `Callback.on_pretrain_routine_end` hook was removed in v1.8."):
        trainer.fit(model)


def test_v2_0_0_deprecated_mc_save_checkpoint():
    mc = ModelCheckpoint()
    trainer = Trainer()
    with mock.patch.object(trainer, "save_checkpoint"), pytest.raises(
        NotImplementedError,
        match=r"ModelCheckpoint.save_checkpoint\(\)` was deprecated in v1.6 and is no longer supported as of 1.8.",
    ):
        mc.save_checkpoint(trainer)
