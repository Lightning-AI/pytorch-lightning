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
r"""
Lambda Callback
^^^^^^^^^^^^^^^

Create a simple callback on the fly using lambda functions.

"""

from typing import Callable, Optional

from pytorch_lightning.callbacks.base import Callback


class LambdaCallback(Callback):
    r"""
    Create a simple callback on the fly using lambda functions.

    Args:
        **kwargs: hooks supported by :class:`~pytorch_lightning.callbacks.base.Callback`

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import LambdaCallback
        >>> trainer = Trainer(callbacks=[LambdaCallback(setup=lambda *args: print('setup'))])
    """

    def __init__(
        self,
        on_before_accelerator_backend_setup: Optional[Callable] = None,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
        on_init_start: Optional[Callable] = None,
        on_init_end: Optional[Callable] = None,
        on_fit_start: Optional[Callable] = None,
        on_fit_end: Optional[Callable] = None,
        on_sanity_check_start: Optional[Callable] = None,
        on_sanity_check_end: Optional[Callable] = None,
        on_train_batch_start: Optional[Callable] = None,
        on_train_batch_end: Optional[Callable] = None,
        on_train_epoch_start: Optional[Callable] = None,
        on_train_epoch_end: Optional[Callable] = None,
        on_validation_epoch_start: Optional[Callable] = None,
        on_validation_epoch_end: Optional[Callable] = None,
        on_test_epoch_start: Optional[Callable] = None,
        on_test_epoch_end: Optional[Callable] = None,
        on_epoch_start: Optional[Callable] = None,
        on_epoch_end: Optional[Callable] = None,
        on_batch_start: Optional[Callable] = None,
        on_validation_batch_start: Optional[Callable] = None,
        on_validation_batch_end: Optional[Callable] = None,
        on_test_batch_start: Optional[Callable] = None,
        on_test_batch_end: Optional[Callable] = None,
        on_batch_end: Optional[Callable] = None,
        on_train_start: Optional[Callable] = None,
        on_train_end: Optional[Callable] = None,
        on_pretrain_routine_start: Optional[Callable] = None,
        on_pretrain_routine_end: Optional[Callable] = None,
        on_validation_start: Optional[Callable] = None,
        on_validation_end: Optional[Callable] = None,
        on_test_start: Optional[Callable] = None,
        on_test_end: Optional[Callable] = None,
        on_keyboard_interrupt: Optional[Callable] = None,
        on_save_checkpoint: Optional[Callable] = None,
        on_load_checkpoint: Optional[Callable] = None,
        on_after_backward: Optional[Callable] = None,
        on_before_zero_grad: Optional[Callable] = None,
    ):
        if on_before_accelerator_backend_setup is not None:
            self.on_before_accelerator_backend_setup = on_before_accelerator_backend_setup
        if setup is not None:
            self.setup = setup
        if teardown is not None:
            self.teardown = teardown
        if on_init_start is not None:
            self.on_init_start = on_init_start
        if on_init_end is not None:
            self.on_init_end = on_init_end
        if on_fit_start is not None:
            self.on_fit_start = on_fit_start
        if on_fit_end is not None:
            self.on_fit_end = on_fit_end
        if on_sanity_check_start is not None:
            self.on_sanity_check_start = on_sanity_check_start
        if on_sanity_check_end is not None:
            self.on_sanity_check_end = on_sanity_check_end
        if on_train_batch_start is not None:
            self.on_train_batch_start = on_train_batch_start
        if on_train_batch_end is not None:
            self.on_train_batch_end = on_train_batch_end
        if on_train_epoch_start is not None:
            self.on_train_epoch_start = on_train_epoch_start
        if on_train_epoch_end is not None:
            self.on_train_epoch_end = on_train_epoch_end
        if on_validation_epoch_start is not None:
            self.on_validation_epoch_start = on_validation_epoch_start
        if on_validation_epoch_end is not None:
            self.on_validation_epoch_end = on_validation_epoch_end
        if on_test_epoch_start is not None:
            self.on_test_epoch_start = on_test_epoch_start
        if on_test_epoch_end is not None:
            self.on_test_epoch_end = on_test_epoch_end
        if on_epoch_start is not None:
            self.on_epoch_start = on_epoch_start
        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        if on_batch_start is not None:
            self.on_batch_start = on_batch_start
        if on_validation_batch_start is not None:
            self.on_validation_batch_start = on_validation_batch_start
        if on_validation_batch_end is not None:
            self.on_validation_batch_end = on_validation_batch_end
        if on_test_batch_start is not None:
            self.on_test_batch_start = on_test_batch_start
        if on_test_batch_end is not None:
            self.on_test_batch_end = on_test_batch_end
        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        if on_train_start is not None:
            self.on_train_start = on_train_start
        if on_train_end is not None:
            self.on_train_end = on_train_end
        if on_pretrain_routine_start is not None:
            self.on_pretrain_routine_start = on_pretrain_routine_start
        if on_pretrain_routine_end is not None:
            self.on_pretrain_routine_end = on_pretrain_routine_end
        if on_validation_start is not None:
            self.on_validation_start = on_validation_start
        if on_validation_end is not None:
            self.on_validation_end = on_validation_end
        if on_test_start is not None:
            self.on_test_start = on_test_start
        if on_test_end is not None:
            self.on_test_end = on_test_end
        if on_keyboard_interrupt is not None:
            self.on_keyboard_interrupt = on_keyboard_interrupt
        if on_save_checkpoint is not None:
            self.on_save_checkpoint = on_save_checkpoint
        if on_load_checkpoint is not None:
            self.on_load_checkpoint = on_load_checkpoint
        if on_after_backward is not None:
            self.on_after_backward = on_after_backward
        if on_before_zero_grad is not None:
            self.on_before_zero_grad = on_before_zero_grad
