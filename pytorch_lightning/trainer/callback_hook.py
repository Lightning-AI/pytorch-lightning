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

from abc import ABC
from copy import deepcopy
from typing import Callable, List

from pytorch_lightning.callbacks import Callback


class TrainerCallbackHookMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    # the proper values/initialisation should be done in child class
    callbacks: List[Callback] = []
    get_model: Callable

    def setup(self, model, stage: str):
        """Called in the beginning of fit and test"""
        for callback in self.callbacks:
            callback.setup(self, model, stage)

    def teardown(self, stage: str):
        """Called at the end of fit and test"""
        for callback in self.callbacks:
            callback.teardown(self, self.get_model(), stage)

    def on_init_start(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_init_start(self)

    def on_init_end(self):
        """Called when the trainer initialization ends, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_init_end(self)

    def on_fit_start(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_fit_start(self, self.get_model())

    def on_fit_end(self):
        """Called when the trainer initialization begins, model has not yet been set."""
        for callback in self.callbacks:
            callback.on_fit_end(self, self.get_model())

    def on_sanity_check_start(self):
        """Called when the validation sanity check starts."""
        for callback in self.callbacks:
            callback.on_sanity_check_start(self, self.get_model())

    def on_sanity_check_end(self):
        """Called when the validation sanity check ends."""
        for callback in self.callbacks:
            callback.on_sanity_check_end(self, self.get_model())

    def on_train_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_train_epoch_start(self, self.get_model())

    def on_train_epoch_end(self, outputs):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_train_epoch_end(self, self.get_model(), outputs)

    def on_validation_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_validation_epoch_start(self, self.get_model())

    def on_validation_epoch_end(self):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_validation_epoch_end(self, self.get_model())

    def on_test_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_test_epoch_start(self, self.get_model())

    def on_test_epoch_end(self):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_test_epoch_end(self, self.get_model())

    def on_epoch_start(self):
        """Called when the epoch begins."""
        for callback in self.callbacks:
            callback.on_epoch_start(self, self.get_model())

    def on_epoch_end(self):
        """Called when the epoch ends."""
        for callback in self.callbacks:
            callback.on_epoch_end(self, self.get_model())

    def on_train_start(self):
        """Called when the train begins."""
        for callback in self.callbacks:
            callback.on_train_start(self, self.get_model())

    def on_train_end(self):
        """Called when the train ends."""
        for callback in self.callbacks:
            callback.on_train_end(self, self.get_model())

    def on_pretrain_routine_start(self, model):
        """Called when the train begins."""
        for callback in self.callbacks:
            callback.on_pretrain_routine_start(self, model)

    def on_pretrain_routine_end(self, model):
        """Called when the train ends."""
        for callback in self.callbacks:
            callback.on_pretrain_routine_end(self, model)

    def on_batch_start(self):
        """Called when the training batch begins."""
        for callback in self.callbacks:
            callback.on_batch_start(self, self.get_model())

    def on_batch_end(self):
        """Called when the training batch ends."""
        for callback in self.callbacks:
            callback.on_batch_end(self, self.get_model())

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        """Called when the training batch begins."""
        for callback in self.callbacks:
            callback.on_train_batch_start(self, self.get_model(), batch, batch_idx, dataloader_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Called when the training batch ends."""
        for callback in self.callbacks:
            callback.on_train_batch_end(self, self.get_model(), outputs, batch, batch_idx, dataloader_idx)

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        """Called when the validation batch begins."""
        for callback in self.callbacks:
            callback.on_validation_batch_start(self, self.get_model(), batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        for callback in self.callbacks:
            callback.on_validation_batch_end(self, self.get_model(), outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        """Called when the test batch begins."""
        for callback in self.callbacks:
            callback.on_test_batch_start(self, self.get_model(), batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        for callback in self.callbacks:
            callback.on_test_batch_end(self, self.get_model(), outputs, batch, batch_idx, dataloader_idx)

    def on_validation_start(self):
        """Called when the validation loop begins."""
        for callback in self.callbacks:
            callback.on_validation_start(self, self.get_model())

    def on_validation_end(self):
        """Called when the validation loop ends."""
        for callback in self.callbacks:
            callback.on_validation_end(self, self.get_model())

    def on_test_start(self):
        """Called when the test begins."""
        for callback in self.callbacks:
            callback.on_test_start(self, self.get_model())

    def on_test_end(self):
        """Called when the test ends."""
        for callback in self.callbacks:
            callback.on_test_end(self, self.get_model())

    def on_keyboard_interrupt(self):
        """Called when the training is interrupted by KeyboardInterrupt."""
        for callback in self.callbacks:
            callback.on_keyboard_interrupt(self, self.get_model())

    def on_save_checkpoint(self):
        """Called when saving a model checkpoint."""
        callback_states = {}
        for callback in self.callbacks:
            callback_class = type(callback)
            state = callback.on_save_checkpoint(self, self.get_model())
            if state:
                callback_states[callback_class] = state
        return callback_states

    def on_load_checkpoint(self, checkpoint):
        """Called when loading a model checkpoint."""
        callback_states = checkpoint.get('callbacks')
        for callback in self.callbacks:
            state = callback_states.get(type(callback))
            if state:
                state = deepcopy(state)
                callback.on_load_checkpoint(state)

    def on_after_backward(self):
        """
        Called after loss.backward() and before optimizers do anything.
        """
        for callback in self.callbacks:
            callback.on_after_backward(self, self.get_model())

    def on_before_zero_grad(self, optimizer):
        """
        Called after optimizer.step() and before optimizer.zero_grad().
        """
        for callback in self.callbacks:
            callback.on_before_zero_grad(self, self.get_model(), optimizer)
