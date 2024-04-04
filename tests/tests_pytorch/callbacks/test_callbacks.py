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
from pathlib import Path
from re import escape
from unittest.mock import Mock

import pytest
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning_utilities.test.warning import no_warning_call


def test_callbacks_configured_in_model(tmp_path):
    """Test the callback system with callbacks added through the model hook."""
    model_callback_mock = Mock(spec=Callback, model=Callback())
    trainer_callback_mock = Mock(spec=Callback, model=Callback())

    class TestModel(BoringModel):
        def configure_callbacks(self):
            return [model_callback_mock]

    model = TestModel()
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_checkpointing": False,
        "fast_dev_run": True,
        "enable_progress_bar": False,
    }

    def assert_expected_calls(_trainer, model_callback, trainer_callback):
        # assert that the rest of calls are the same as for trainer callbacks
        expected_calls = [m for m in trainer_callback.method_calls if m]
        assert expected_calls
        assert model_callback.method_calls == expected_calls

    # .fit()
    trainer_options.update(callbacks=[trainer_callback_mock])
    trainer = Trainer(**trainer_options)

    assert trainer_callback_mock in trainer.callbacks
    assert model_callback_mock not in trainer.callbacks
    trainer.fit(model)

    assert model_callback_mock in trainer.callbacks
    assert trainer.callbacks[-1] == model_callback_mock
    assert_expected_calls(trainer, model_callback_mock, trainer_callback_mock)

    # .test()
    for fn in ("test", "validate"):
        model_callback_mock.reset_mock()
        trainer_callback_mock.reset_mock()

        trainer_options.update(callbacks=[trainer_callback_mock])
        trainer = Trainer(**trainer_options)

        trainer_fn = getattr(trainer, fn)
        trainer_fn(model)

        assert model_callback_mock in trainer.callbacks
        assert trainer.callbacks[-1] == model_callback_mock
        assert_expected_calls(trainer, model_callback_mock, trainer_callback_mock)


def test_configure_callbacks_hook_multiple_calls(tmp_path):
    """Test that subsequent calls to `configure_callbacks` do not change the callbacks list."""
    model_callback_mock = Mock(spec=Callback, model=Callback())

    class TestModel(BoringModel):
        def configure_callbacks(self):
            return model_callback_mock

    model = TestModel()
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True, enable_checkpointing=False)

    callbacks_before_fit = trainer.callbacks.copy()
    assert callbacks_before_fit

    trainer.fit(model)
    callbacks_after_fit = trainer.callbacks.copy()
    assert callbacks_after_fit == callbacks_before_fit + [model_callback_mock]

    for fn in ("test", "validate"):
        trainer_fn = getattr(trainer, fn)
        trainer_fn(model)

        callbacks_after = trainer.callbacks.copy()
        assert callbacks_after == callbacks_after_fit

        trainer_fn(model)
        callbacks_after = trainer.callbacks.copy()
        assert callbacks_after == callbacks_after_fit


class OldStatefulCallback(Callback):
    def __init__(self, state):
        self.state = state

    @property
    def state_key(self):
        return type(self)

    def state_dict(self):
        return {"state": self.state}

    def load_state_dict(self, state_dict) -> None:
        self.state = state_dict["state"]


def test_resume_callback_state_saved_by_type_stateful(tmp_path):
    """Test that a legacy checkpoint that didn't use a state key before can still be loaded, using
    state_dict/load_state_dict."""
    model = BoringModel()
    callback = OldStatefulCallback(state=111)
    trainer = Trainer(default_root_dir=tmp_path, max_steps=1, callbacks=[callback])
    trainer.fit(model)
    ckpt_path = Path(trainer.checkpoint_callback.best_model_path)
    assert ckpt_path.exists()

    callback = OldStatefulCallback(state=222)
    trainer = Trainer(default_root_dir=tmp_path, max_steps=2, callbacks=[callback])
    trainer.fit(model, ckpt_path=ckpt_path)
    assert callback.state == 111


def test_resume_incomplete_callbacks_list_warning(tmp_path):
    model = BoringModel()
    callback0 = ModelCheckpoint(monitor="epoch")
    callback1 = ModelCheckpoint(monitor="global_step")
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        callbacks=[callback0, callback1],
    )
    trainer.fit(model)
    ckpt_path = trainer.checkpoint_callback.best_model_path

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        callbacks=[callback1],  # one callback is missing!
    )
    with pytest.warns(UserWarning, match=escape(f"Please add the following callbacks: [{repr(callback0.state_key)}]")):
        trainer.fit(model, ckpt_path=ckpt_path)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        callbacks=[callback1, callback0],  # all callbacks here, order switched
    )
    with no_warning_call(UserWarning, match="Please add the following callbacks:"):
        trainer.fit(model, ckpt_path=ckpt_path)
