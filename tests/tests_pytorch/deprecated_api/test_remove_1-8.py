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


def test_v_1_8_0_deprecated_device_stats_monitor_prefix_metric_keys():
    from pytorch_lightning.callbacks.device_stats_monitor import prefix_metric_keys

    with pytest.deprecated_call(match="in v1.6 and will be removed in v1.8"):
        prefix_metric_keys({"foo": 1.0}, "bar")


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
