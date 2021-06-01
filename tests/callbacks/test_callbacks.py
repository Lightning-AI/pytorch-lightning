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
from unittest import mock
from unittest.mock import ANY, call, MagicMock, Mock

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


@mock.patch("torch.save")  # need to mock torch.save or we get pickle error
def test_trainer_callback_hook_system_fit(_, tmpdir):
    """Test the callback hook system for fit."""

    model = BoringModel()
    callback_mock = MagicMock()
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[callback_mock],
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=3,
        progress_bar_refresh_rate=0,
    )

    # check that only the to calls exists
    assert trainer.callbacks[0] == callback_mock
    assert callback_mock.method_calls == [
        call.on_init_start(trainer),
        call.on_init_end(trainer),
    ]

    # fit model
    trainer.fit(model)

    assert callback_mock.method_calls == [
        call.on_init_start(trainer),
        call.on_init_end(trainer),
        call.on_before_accelerator_backend_setup(trainer, model),
        call.setup(trainer, model, 'fit'),
        call.on_configure_sharded_model(trainer, model),
        call.on_fit_start(trainer, model),
        call.on_pretrain_routine_start(trainer, model),
        call.on_pretrain_routine_end(trainer, model),
        call.on_sanity_check_start(trainer, model),
        call.on_validation_start(trainer, model),
        call.on_epoch_start(trainer, model),
        call.on_validation_epoch_start(trainer, model),
        call.on_validation_batch_start(trainer, model, ANY, 0, 0),
        call.on_validation_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_validation_epoch_end(trainer, model),
        call.on_epoch_end(trainer, model),
        call.on_validation_end(trainer, model),
        call.on_sanity_check_end(trainer, model),
        call.on_train_start(trainer, model),
        call.on_epoch_start(trainer, model),
        call.on_train_epoch_start(trainer, model),
        call.on_batch_start(trainer, model),
        call.on_train_batch_start(trainer, model, ANY, 0, 0),
        call.on_before_zero_grad(trainer, model, trainer.optimizers[0]),
        call.on_after_backward(trainer, model),
        call.on_train_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_batch_end(trainer, model),
        call.on_batch_start(trainer, model),
        call.on_train_batch_start(trainer, model, ANY, 1, 0),
        call.on_before_zero_grad(trainer, model, trainer.optimizers[0]),
        call.on_after_backward(trainer, model),
        call.on_train_batch_end(trainer, model, ANY, ANY, 1, 0),
        call.on_batch_end(trainer, model),
        call.on_batch_start(trainer, model),
        call.on_train_batch_start(trainer, model, ANY, 2, 0),
        call.on_before_zero_grad(trainer, model, trainer.optimizers[0]),
        call.on_after_backward(trainer, model),
        call.on_train_batch_end(trainer, model, ANY, ANY, 2, 0),
        call.on_batch_end(trainer, model),
        call.on_validation_start(trainer, model),
        call.on_epoch_start(trainer, model),
        call.on_validation_epoch_start(trainer, model),
        call.on_validation_batch_start(trainer, model, ANY, 0, 0),
        call.on_validation_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_validation_epoch_end(trainer, model),
        call.on_epoch_end(trainer, model),
        call.on_validation_end(trainer, model),
        call.on_save_checkpoint(trainer, model),  # should take ANY but we are inspecting signature for BC
        call.on_train_epoch_end(trainer, model, ANY),
        call.on_epoch_end(trainer, model),
        call.on_train_end(trainer, model),
        call.on_fit_end(trainer, model),
        call.teardown(trainer, model, 'fit'),
    ]


def test_trainer_callback_hook_system_test(tmpdir):
    """Test the callback hook system for test."""

    model = BoringModel()
    callback_mock = MagicMock()
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[callback_mock],
        max_epochs=1,
        limit_test_batches=2,
        progress_bar_refresh_rate=0,
    )

    trainer.test(model)

    assert callback_mock.method_calls == [
        call.on_init_start(trainer),
        call.on_init_end(trainer),
        call.on_before_accelerator_backend_setup(trainer, model),
        call.setup(trainer, model, 'test'),
        call.on_configure_sharded_model(trainer, model),
        call.on_test_start(trainer, model),
        call.on_epoch_start(trainer, model),
        call.on_test_epoch_start(trainer, model),
        call.on_test_batch_start(trainer, model, ANY, 0, 0),
        call.on_test_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_test_batch_start(trainer, model, ANY, 1, 0),
        call.on_test_batch_end(trainer, model, ANY, ANY, 1, 0),
        call.on_test_epoch_end(trainer, model),
        call.on_epoch_end(trainer, model),
        call.on_test_end(trainer, model),
        call.teardown(trainer, model, 'test'),
    ]


def test_trainer_callback_hook_system_validate(tmpdir):
    """Test the callback hook system for validate."""

    model = BoringModel()
    callback_mock = MagicMock()
    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[callback_mock],
        max_epochs=1,
        limit_val_batches=2,
        progress_bar_refresh_rate=0,
    )

    trainer.validate(model)

    assert callback_mock.method_calls == [
        call.on_init_start(trainer),
        call.on_init_end(trainer),
        call.on_before_accelerator_backend_setup(trainer, model),
        call.setup(trainer, model, 'validate'),
        call.on_configure_sharded_model(trainer, model),
        call.on_validation_start(trainer, model),
        call.on_epoch_start(trainer, model),
        call.on_validation_epoch_start(trainer, model),
        call.on_validation_batch_start(trainer, model, ANY, 0, 0),
        call.on_validation_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_validation_batch_start(trainer, model, ANY, 1, 0),
        call.on_validation_batch_end(trainer, model, ANY, ANY, 1, 0),
        call.on_validation_epoch_end(trainer, model),
        call.on_epoch_end(trainer, model),
        call.on_validation_end(trainer, model),
        call.teardown(trainer, model, 'validate'),
    ]


# TODO: add callback tests for predict and tune


def test_callbacks_configured_in_model(tmpdir):
    """ Test the callback system with callbacks added through the model hook. """

    model_callback_mock = Mock()
    trainer_callback_mock = Mock()

    class TestModel(BoringModel):

        def configure_callbacks(self):
            return [model_callback_mock]

    model = TestModel()
    trainer_options = dict(
        default_root_dir=tmpdir,
        checkpoint_callback=False,
        fast_dev_run=True,
        progress_bar_refresh_rate=0,
    )

    def assert_expected_calls(_trainer, model_callback, trainer_callback):
        # some methods in callbacks configured through model won't get called
        uncalled_methods = [
            call.on_init_start(_trainer),
            call.on_init_end(_trainer),
        ]
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
    """ Test that subsequent calls to `configure_callbacks` do not change the callbacks list. """
    model_callback_mock = Mock()

    class TestModel(BoringModel):

        def configure_callbacks(self):
            return [model_callback_mock]

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        checkpoint_callback=False,
        progress_bar_refresh_rate=1,
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
