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
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, LightningModule
from tests.base import BoringModel
from unittest.mock import MagicMock, patch, ANY, call

from pytorch_lightning import Trainer
from tests.base import BoringModel

@patch("torch.save")  # need to mock torch.save or we get pickle error
def test_callback_system(torch_save):
    model = BoringModel()
    # pretend to be a callback, record all calls
    callback = MagicMock()
    trainer = Trainer(callbacks=[callback], max_epochs=1, num_sanity_val_steps=1,
                      limit_train_batches=1, limit_val_batches=1, limit_test_batches=1)
    trainer.fit(model)

    # check if a method was called exactly once
    callback.on_fit_start.assert_called_once()

    # check how many times a method was called
    assert callback.on_train_batch_end.call_count == 1

    # check that a method was NEVER called
    callback.on_keyboard_interrupt.assert_not_called()

    # check with what a method was called
    callback.on_fit_end.assert_called_with(trainer, model)

    # check exact call order
    callback.assert_has_calls([
        call.on_init_start(trainer),
        call.on_init_end(trainer),
        call.setup(trainer, None, "fit"),
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
        # # sanity check end

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

    test_callback = MagicMock()
    trainer = Trainer(callbacks=[test_callback], limit_test_batches=1, max_epochs=1, num_sanity_val_steps=0)
    trainer.test(model)

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
