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
Subclass this class and override any of the relevant hooks

"""

import abc


class Callback(abc.ABC):
    r"""
    Abstract base class used to build new callbacks.
    """

    def setup(self, trainer, pl_module, stage: str):
        """Called when fit or test begins"""
        pass

    def teardown(self, trainer, pl_module, stage: str):
        """Called when fit or test ends"""
        pass

    def on_init_start(self, trainer):
        """Called when the trainer initialization begins, model has not yet been set."""
        pass

    def on_init_end(self, trainer):
        """Called when the trainer initialization ends, model has not yet been set."""
        pass

    def on_fit_start(self, trainer, pl_module):
        """Called when fit begins"""
        pass

    def on_fit_end(self, trainer, pl_module):
        """Called when fit ends"""
        pass

    def on_sanity_check_start(self, trainer, pl_module):
        """Called when the validation sanity check starts."""
        pass

    def on_sanity_check_end(self, trainer, pl_module):
        """Called when the validation sanity check ends."""
        pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the train batch begins."""
        pass

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the train batch ends."""
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""
        pass

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the val epoch begins."""
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        pass

    def on_test_epoch_start(self, trainer, pl_module):
        """Called when the test epoch begins."""
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        pass

    def on_epoch_start(self, trainer, pl_module):
        """Called when the epoch begins."""
        pass

    def on_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""
        pass

    def on_batch_start(self, trainer, pl_module):
        """Called when the training batch begins."""
        pass

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the validation batch begins."""
        pass

    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        pass

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the test batch begins."""
        pass

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        pass

    def on_batch_end(self, trainer, pl_module):
        """Called when the training batch ends."""
        pass

    def on_train_start(self, trainer, pl_module):
        """Called when the train begins."""
        pass

    def on_train_end(self, trainer, pl_module):
        """Called when the train ends."""
        pass

    def on_pretrain_routine_start(self, trainer, pl_module):
        """Called when the pretrain routine begins."""
        pass

    def on_pretrain_routine_end(self, trainer, pl_module):
        """Called when the pretrain routine ends."""
        pass

    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, trainer, pl_module):
        """Called when the test begins."""
        pass

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        pass

    def on_keyboard_interrupt(self, trainer, pl_module):
        """Called when the training is interrupted by KeyboardInterrupt."""
