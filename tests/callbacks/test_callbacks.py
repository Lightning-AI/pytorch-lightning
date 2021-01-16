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
from unittest.mock import ANY, call, MagicMock

from pytorch_lightning import Trainer
from tests.base import BoringModel


@mock.patch("torch.save")  # need to mock torch.save or we get pickle error
def test_trainer_callback_system(torch_save):
    """Test the callback system."""

    model = BoringModel()

    callback_mock = MagicMock()

    trainer_options = dict(
        callbacks=[callback_mock],
        max_epochs=1,
        limit_val_batches=1,
        limit_train_batches=3,
        limit_test_batches=2,
        progress_bar_refresh_rate=0,
    )

    # no call yet
    callback_mock.assert_not_called()

    # fit model
    trainer = Trainer(**trainer_options)

    # check that only the to calls exists
    assert trainer.callbacks[0] == callback_mock
    assert callback_mock.method_calls == [
        call.on_init_start(trainer),
        call.on_init_end(trainer),
    ]

    trainer.fit(model)

    assert callback_mock.method_calls == [
        call.on_init_start(trainer),
        call.on_init_end(trainer),
        call.setup(trainer, model, 'fit'),
        call.on_fit_start(trainer, model),
        call.on_pretrain_routine_start(trainer, model),
        call.on_pretrain_routine_end(trainer, model),
        call.on_sanity_check_start(trainer, model),
        call.on_validation_start(trainer, model),
        call.on_validation_epoch_start(trainer, model),
        call.on_validation_batch_start(trainer, model, ANY, 0, 0),
        call.on_validation_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_validation_epoch_end(trainer, model),
        call.on_validation_end(trainer, model),
        call.on_sanity_check_end(trainer, model),
        call.on_train_start(trainer, model),
        call.on_epoch_start(trainer, model),
        call.on_train_epoch_start(trainer, model),
        call.on_batch_start(trainer, model),
        call.on_train_batch_start(trainer, model, ANY, 0, 0),
        call.on_after_backward(trainer, model),
        call.on_before_zero_grad(trainer, model, trainer.optimizers[0]),
        call.on_batch_end(trainer, model),
        call.on_train_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_batch_start(trainer, model),
        call.on_train_batch_start(trainer, model, ANY, 1, 0),
        call.on_after_backward(trainer, model),
        call.on_before_zero_grad(trainer, model, trainer.optimizers[0]),
        call.on_batch_end(trainer, model),
        call.on_train_batch_end(trainer, model, ANY, ANY, 1, 0),
        call.on_batch_start(trainer, model),
        call.on_train_batch_start(trainer, model, ANY, 2, 0),
        call.on_after_backward(trainer, model),
        call.on_before_zero_grad(trainer, model, trainer.optimizers[0]),
        call.on_batch_end(trainer, model),
        call.on_train_batch_end(trainer, model, ANY, ANY, 2, 0),
        call.on_validation_start(trainer, model),
        call.on_validation_epoch_start(trainer, model),
        call.on_validation_batch_start(trainer, model, ANY, 0, 0),
        call.on_validation_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_validation_epoch_end(trainer, model),
        call.on_validation_end(trainer, model),
        call.on_save_checkpoint(trainer, model),
        call.on_epoch_end(trainer, model),
        call.on_train_epoch_end(trainer, model, ANY),
        call.on_train_end(trainer, model),
        call.on_fit_end(trainer, model),
        call.teardown(trainer, model, 'fit'),
    ]

    callback_mock.reset_mock()
    trainer = Trainer(**trainer_options)
    trainer.test(model)

    assert callback_mock.method_calls == [
        call.on_init_start(trainer),
        call.on_init_end(trainer),
        call.setup(trainer, model, 'test'),
        call.on_fit_start(trainer, model),
        call.on_test_start(trainer, model),
        call.on_test_epoch_start(trainer, model),
        call.on_test_batch_start(trainer, model, ANY, 0, 0),
        call.on_test_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_test_batch_start(trainer, model, ANY, 1, 0),
        call.on_test_batch_end(trainer, model, ANY, ANY, 1, 0),
        call.on_test_epoch_end(trainer, model),
        call.on_test_end(trainer, model),
        call.on_fit_end(trainer, model),
        call.teardown(trainer, model, 'fit'),
        call.teardown(trainer, model, 'test'),
    ]
