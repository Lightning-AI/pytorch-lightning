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
from typing import Any, Dict, List, Type, Union

from packaging.version import Version

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import STEP_OUTPUT


class TrainerCallbackHookMixin(ABC):

    # this is just a summary on variables used in this abstract class,
    # the proper values/initialisation should be done in child class
    callbacks: List[Callback] = []
    lightning_module: "pl.LightningModule"

    # TODO: Update this in v1.7 (deprecation: #9816)
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """Called when the training batch begins."""
        for callback in self.callbacks:
            if is_param_in_hook_signature(callback.on_train_batch_start, "dataloader_idx", explicit=True):
                callback.on_train_batch_start(self, self.lightning_module, batch, batch_idx, 0)
            else:
                callback.on_train_batch_start(self, self.lightning_module, batch, batch_idx)

    # TODO: Update this in v1.7 (deprecation: #9816)
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch, batch_idx, dataloader_idx=0):
        """Called when the training batch ends."""
        for callback in self.callbacks:
            if is_param_in_hook_signature(callback.on_train_batch_end, "dataloader_idx", explicit=True):
                callback.on_train_batch_end(self, self.lightning_module, outputs, batch, batch_idx, 0)
            else:
                callback.on_train_batch_end(self, self.lightning_module, outputs, batch, batch_idx)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, dict]:
        """Called when saving a model checkpoint."""
        callback_states = {}
        for callback in self.callbacks:
            state = callback.on_save_checkpoint(self, self.lightning_module, checkpoint)
            if state:
                callback_states[callback.state_key] = state
        return callback_states

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a model checkpoint."""
        # Todo: the `callback_states` are dropped with TPUSpawn as they
        # can't be saved using `xm.save`
        # https://github.com/pytorch/xla/issues/2773
        callback_states: Dict[Union[Type, str], Dict] = checkpoint.get("callbacks")

        if callback_states is None:
            return

        is_legacy_ckpt = Version(checkpoint["pytorch-lightning_version"]) < Version("1.5.0dev")
        current_callbacks_keys = {cb._legacy_state_key if is_legacy_ckpt else cb.state_key for cb in self.callbacks}
        difference = callback_states.keys() - current_callbacks_keys
        if difference:
            rank_zero_warn(
                "Be aware that when using `ckpt_path`,"
                " callbacks used to create the checkpoint need to be provided during `Trainer` instantiation."
                f" Please add the following callbacks: {list(difference)}.",
                UserWarning,
            )

        for callback in self.callbacks:
            state = callback_states.get(callback.state_key, callback_states.get(callback._legacy_state_key))
            if state:
                state = deepcopy(state)
                callback.on_load_checkpoint(self, self.lightning_module, state)
