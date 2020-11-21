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
from unittest.mock import ANY, MagicMock, call, patch

from pytorch_lightning import Callback, LightningModule, Trainer
from tests.base import BoringModel

from pytorch_lightning import Trainer
from tests.base import BoringModel

@patch("torch.save")  # need to mock torch.save or we get pickle error
def test_callback_hooks(torch_save):
    """
    Tests that callback methods (exclude on_load_checkpoint)
    - are called in order
    - are not called
    - are called exactly once
    - are called with arguments
    - are called how many times
    """

    model = BoringModel()
    # pretend to be a callback, record all calls
    callback = MagicMock()
    trainer = Trainer(callbacks=[callback], max_epochs=1, num_sanity_val_steps=1,
                      limit_train_batches=1, limit_val_batches=1, limit_test_batches=1)
    trainer.fit(model)

    # check how many times a method was called
    assert callback.on_init_start.call_count == 1
    assert callback.on_init_end.call_count == 1

    assert callback.setup.call_count == 1

    assert callback.on_fit_start.call_count == 1
    assert callback.on_pretrain_routine_start.call_count == 1
    assert callback.on_pretrain_routine_end.call_count == 1

    assert callback.on_sanity_check_start.call_count == 1

    assert callback.on_validation_start.call_count == 2
    assert callback.on_validation_epoch_start.call_count == 2
    assert callback.on_validation_batch_start.call_count == 2
    assert callback.on_validation_batch_end.call_count == 2
    assert callback.on_validation_epoch_end.call_count == 2
    assert callback.on_validation_end.call_count == 2

    assert callback.on_sanity_check_end.call_count == 1

    assert callback.on_train_start.call_count == 1

    assert callback.on_epoch_start.call_count == 1
    assert callback.on_train_epoch_start.call_count == 1

    assert callback.on_batch_start.call_count == 1
    assert callback.on_train_batch_start.call_count == 1

    assert callback.on_after_backward.call_count == 1
    assert callback.on_before_zero_grad.call_count == 1

    assert callback.on_batch_end.call_count == 1
    assert callback.on_train_batch_end.call_count == 1

    assert callback.on_save_checkpoint.call_count == 1

    assert callback.on_epoch_end.call_count == 1
    assert callback.on_train_epoch_end.call_count == 1

    assert callback.on_train_end.call_count == 1

    assert callback.on_fit_end.call_count == 1
    assert callback.teardown.call_count == 1

    # check if a method was called exactly once
    callback.on_init_start.assert_called_once()
    callback.on_init_end.assert_called_once()

    callback.setup.assert_called_once()

    callback.on_fit_start.assert_called_once()
    callback.on_pretrain_routine_start.assert_called_once()
    callback.on_pretrain_routine_end.assert_called_once()

    callback.on_sanity_check_start.assert_called_once()
    callback.on_sanity_check_end.assert_called_once()

    callback.on_train_start.assert_called_once()

    callback.on_epoch_start.assert_called_once()
    callback.on_train_epoch_start.assert_called_once()

    callback.on_batch_start.assert_called_once()
    callback.on_train_batch_start.assert_called_once()

    callback.on_after_backward.assert_called_once()
    callback.on_before_zero_grad.assert_called_once()

    callback.on_batch_end.assert_called_once()
    callback.on_train_batch_end.assert_called_once()

    callback.on_save_checkpoint.assert_called_once()

    callback.on_epoch_end.assert_called_once()
    callback.on_train_epoch_end.assert_called_once()

    callback.on_train_end.assert_called_once()

    callback.on_fit_end.assert_called_once()
    callback.teardown.assert_called_once()

    # check that a method was NEVER called
    callback.on_keyboard_interrupt.assert_not_called()
    callback.on_test_start.assert_not_called()
    callback.on_test_epoch_start.assert_not_called()
    callback.on_test_batch_start.assert_not_called()
    callback.on_test_batch_end.assert_not_called()
    callback.on_test_epoch_end.assert_not_called()
    callback.on_test_end.assert_not_called()

    # check with what a method was called
    callback.on_init_start.assert_called_with(trainer)
    callback.on_init_end.assert_called_with(trainer)

    callback.setup.assert_called_with(trainer, model, 'fit')

    callback.on_fit_start.assert_called_with(trainer, model)
    callback.on_pretrain_routine_start.assert_called_with(trainer, model)
    callback.on_pretrain_routine_end.assert_called_with(trainer, model)

    callback.on_sanity_check_start.assert_called_with(trainer, model)
    callback.on_validation_start.assert_called_with(trainer, model),
    callback.on_validation_epoch_start.assert_called_with(trainer, model),
    callback.on_validation_batch_start.assert_called_with(trainer, model, ANY, 0, 0),
    callback.on_validation_batch_end.assert_called_with(trainer, model, ANY, ANY, 0, 0),
    callback.on_validation_epoch_end.assert_called_with(trainer, model),
    callback.on_validation_end.assert_called_with(trainer, model),
    callback.on_sanity_check_end.assert_called_with(trainer, model)

    callback.on_train_start.assert_called_with(trainer, model)

    callback.on_epoch_start.assert_called_with(trainer, model)
    callback.on_train_epoch_start.assert_called_with(trainer, model)

    callback.on_batch_start.assert_called_with(trainer, model)
    callback.on_train_batch_start.assert_called_with(trainer, model, ANY, 0, 0)

    callback.on_after_backward.assert_called_with(trainer, model)
    callback.on_before_zero_grad.assert_called_with(trainer, model, ANY)

    callback.on_batch_end.assert_called_with(trainer, model)
    callback.on_train_batch_end.assert_called_with(trainer, model, ANY, ANY, 0, 0)

    callback.on_save_checkpoint.assert_called_with(trainer, model)

    callback.on_epoch_end.assert_called_with(trainer, model)
    callback.on_train_epoch_end.assert_called_with(trainer, model, ANY)

    callback.on_train_end.assert_called_with(trainer, model)

    callback.on_fit_end.assert_called_with(trainer, model)
    callback.teardown.assert_called_with(trainer, model, 'fit')

    # check exact call order
    callback.assert_has_calls([
        call.on_init_start(trainer),
        call.on_init_end(trainer),
        call.setup(trainer, model, "fit"),
        call.on_fit_start(trainer, model),
        call.on_pretrain_routine_start(trainer, model),
        call.on_pretrain_routine_end(trainer, model),

        # sanity check start
        call.on_sanity_check_start(trainer, model),
        call.on_validation_start(trainer, model),
        call.on_validation_epoch_start(trainer, model),
        call.on_validation_batch_start(trainer, model, ANY, 0, 0),
        call.on_validation_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_validation_epoch_end(trainer, model),
        call.on_validation_end(trainer, model),
        call.on_sanity_check_end(trainer, model),
        # sanity check end

        # train start
        call.on_train_start(trainer, model),
        call.on_epoch_start(trainer, model),
        call.on_train_epoch_start(trainer, model),
        # batch 0
        call.on_batch_start(trainer, model),
        # here we don't care about exact values in batch, so we say ANY
        call.on_train_batch_start(trainer, model, ANY, 0, 0),

        # backward & optimizer
        call.on_after_backward(trainer, model),
        call.on_before_zero_grad(trainer, model, ANY),

        call.on_batch_end(trainer, model),
        call.on_train_batch_end(trainer, model, ANY, ANY, 0, 0),

        # validation start
        call.on_validation_start(trainer, model),
        call.on_validation_epoch_start(trainer, model),
        call.on_validation_batch_start(trainer, model, ANY, 0, 0),
        call.on_validation_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_validation_epoch_end(trainer, model),
        call.on_validation_end(trainer, model),
        # validation end

        # ckpt
        call.on_save_checkpoint(trainer, model),
        call.on_save_checkpoint().__bool__(),  # what's this lol?
        call.on_epoch_end(trainer, model),

        call.on_train_epoch_end(trainer, model, ANY),

        call.on_train_end(trainer, model),
        call.on_fit_end(trainer, model),
        call.teardown(trainer, model, "fit"),
    ])

    # .test()
    test_callback = MagicMock()
    trainer = Trainer(callbacks=[test_callback], limit_test_batches=1, max_epochs=1, num_sanity_val_steps=0)
    trainer.test(model)

    # check how many times a method was called
    assert test_callback.on_init_start.call_count == 1
    assert test_callback.on_init_end.call_count == 1
    assert test_callback.setup.call_count == 1
    assert test_callback.on_fit_start.call_count == 1
    assert test_callback.on_pretrain_routine_start.call_count == 1
    assert test_callback.on_pretrain_routine_end.call_count == 1
    assert test_callback.on_test_start.call_count == 1
    assert test_callback.on_test_epoch_start.call_count == 1
    assert test_callback.on_test_batch_start.call_count == 1
    assert test_callback.on_test_batch_end.call_count == 1
    assert test_callback.on_test_epoch_end.call_count == 1
    assert test_callback.on_test_end.call_count == 1
    assert test_callback.on_fit_end.call_count == 1
    assert test_callback.teardown.call_count == 2

    # check methods are called exactly once
    test_callback.on_init_start.assert_called_once()
    test_callback.on_init_end.assert_called_once()
    test_callback.setup.assert_called_once()
    test_callback.on_fit_start.assert_called_once()
    test_callback.on_pretrain_routine_start.assert_called_once()
    test_callback.on_pretrain_routine_end.assert_called_once()
    test_callback.on_test_start.assert_called_once()
    test_callback.on_test_epoch_start.assert_called_once()
    test_callback.on_test_batch_start.assert_called_once()
    test_callback.on_test_batch_end.assert_called_once()
    test_callback.on_test_epoch_end.assert_called_once()
    test_callback.on_test_end.assert_called_once()
    test_callback.on_fit_end.assert_called_once()

    # check that a method was NEVER called
    test_callback.on_keyboard_interrupt.assert_not_called()
    test_callback.on_sanity_check_start.assert_not_called()
    test_callback.on_sanity_check_end.assert_not_called()
    test_callback.on_train_start.assert_not_called()
    test_callback.on_epoch_start.assert_not_called()
    test_callback.on_train_epoch_start.assert_not_called()
    test_callback.on_batch_start.assert_not_called()
    test_callback.on_train_batch_start.assert_not_called()
    test_callback.on_after_backward.assert_not_called()
    test_callback.on_before_zero_grad.assert_not_called()
    test_callback.on_batch_end.assert_not_called()
    test_callback.on_train_batch_end.assert_not_called()
    test_callback.on_save_checkpoint.assert_not_called()
    test_callback.on_epoch_end.assert_not_called()
    test_callback.on_train_epoch_end.assert_not_called()
    test_callback.on_train_end.assert_not_called()

    # check with what a method was called
    test_callback.on_init_start.assert_called_with(trainer)
    test_callback.on_init_end.assert_called_with(trainer)
    test_callback.setup.assert_called_with(trainer, model, 'test')
    test_callback.on_fit_start.assert_called_with(trainer, model)
    test_callback.on_pretrain_routine_start.assert_called_with(trainer, model)
    test_callback.on_pretrain_routine_end.assert_called_with(trainer, model)

    test_callback.on_test_start.assert_called_with(trainer, model)
    test_callback.on_test_epoch_start.assert_called_with(trainer, model)
    test_callback.on_test_batch_start.assert_called_with(trainer, model, ANY, 0, 0)
    test_callback.on_test_batch_end.assert_called_with(trainer, model, ANY, ANY, 0, 0)
    test_callback.on_test_epoch_end.assert_called_with(trainer, model)
    test_callback.on_test_end.assert_called_with(trainer, model)
    test_callback.on_fit_end.assert_called_with(trainer, model)
    test_callback.teardown.assert_called_with(trainer, model, 'test')

    test_callback.assert_has_calls([
        call.on_init_start(trainer),
        call.on_init_end(trainer),
        call.setup(trainer, model, 'test'),
        call.on_fit_start(trainer, model),
        call.on_pretrain_routine_start(trainer, model),
        call.on_pretrain_routine_end(trainer, model),

        # test start
        call.on_test_start(trainer, model),
        call.on_test_epoch_start(trainer, model),
        call.on_test_batch_start(trainer, model, ANY, 0, 0),
        call.on_test_batch_end(trainer, model, ANY, ANY, 0, 0),
        call.on_test_epoch_end(trainer, model),
        call.on_test_end(trainer, model),
        call.on_fit_end(trainer, model),
        call.teardown(trainer, model, 'fit'),
        call.teardown(trainer, model, 'test')
    ])
