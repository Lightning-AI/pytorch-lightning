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
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.loops.progress import _BaseProgress


class _Loop:
    """Basic Loops interface."""

    def __init__(self, trainer: "pl.Trainer") -> None:
        self._restarting = False
        self._loaded_from_state_dict = False
        self.trainer = trainer

    @property
    def restarting(self) -> bool:
        """Whether the state of this loop was reloaded and it needs to restart."""
        return self._restarting

    @restarting.setter
    def restarting(self, restarting: bool) -> None:
        """Connects this loop's restarting value and its children."""
        self._restarting = restarting
        for loop in vars(self).values():
            if isinstance(loop, _Loop):
                loop.restarting = restarting

    def reset_restart_stage(self) -> None:
        pass

    def on_save_checkpoint(self) -> dict:
        """Called when saving a model checkpoint, use to persist loop state.

        Returns:
            The current loop state.

        """
        return {}

    def on_load_checkpoint(self, state_dict: dict) -> None:
        """Called when loading a model checkpoint, use to reload loop state."""

    def state_dict(self, destination: Optional[dict] = None, prefix: str = "") -> dict:
        """The state dict is determined by the state and progress of this loop and all its children.

        Args:
            destination: An existing dictionary to update with this loop's state. By default a new dictionary
                is returned.
            prefix: A prefix for each key in the state dictionary

        """
        if destination is None:
            destination = {}

        destination[prefix + "state_dict"] = self.on_save_checkpoint()

        for k, v in self.__dict__.items():
            key = prefix + k
            if isinstance(v, _BaseProgress):
                destination[key] = v.state_dict()
            elif isinstance(v, _Loop):
                v.state_dict(destination, key + ".")
        return destination

    def load_state_dict(
        self,
        state_dict: dict,
        prefix: str = "",
    ) -> None:
        """Loads the state of this loop and all its children."""
        self._load_from_state_dict(state_dict.copy(), prefix)
        for k, v in self.__dict__.items():
            if isinstance(v, _Loop):
                v.load_state_dict(state_dict.copy(), prefix + k + ".")
        self.restarting = True
        self._loaded_from_state_dict = True

    def _load_from_state_dict(self, state_dict: dict, prefix: str) -> None:
        for k, v in self.__dict__.items():
            key = prefix + k
            if key not in state_dict:
                # compatibility with old checkpoints
                continue
            if isinstance(v, _BaseProgress):
                v.load_state_dict(state_dict[key])
        if prefix + "state_dict" in state_dict:  # compatibility with old checkpoints
            self.on_load_checkpoint(state_dict[prefix + "state_dict"])

    def on_iteration_done(self) -> None:
        self._restarting = False
        self._loaded_from_state_dict = False
        self.reset_restart_stage()
