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
from unittest.mock import call, Mock

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


def test_callbacks_configured_in_model(tmpdir):
    """Test the callback system with callbacks added through the model hook."""

    model_callback_mock = Mock()
    trainer_callback_mock = Mock()

    class TestModel(BoringModel):
        def configure_callbacks(self):
            return [model_callback_mock]

    model = TestModel()
    trainer_options = dict(
        default_root_dir=tmpdir, checkpoint_callback=False, fast_dev_run=True, progress_bar_refresh_rate=0
    )

    def assert_expected_calls(_trainer, model_callback, trainer_callback):
        # some methods in callbacks configured through model won't get called
        uncalled_methods = [call.on_init_start(_trainer), call.on_init_end(_trainer)]
        for uncalled in uncalled_methods:
            assert uncalled not in model_callback.method_calls

        # assert that the rest of calls are the same as for trainer callbacks
        expected_calls = [m for m in trainer_callback.method_calls if m not in uncalled_methods]
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


def test_configure_callbacks_hook_multiple_calls(tmpdir):
    """Test that subsequent calls to `configure_callbacks` do not change the callbacks list."""
    model_callback_mock = Mock()

    class TestModel(BoringModel):
        def configure_callbacks(self):
            return [model_callback_mock]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, checkpoint_callback=False, progress_bar_refresh_rate=1
    )

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

        trainer_fn(ckpt_path=None)
        callbacks_after = trainer.callbacks.copy()
        assert callbacks_after == callbacks_after_fit
