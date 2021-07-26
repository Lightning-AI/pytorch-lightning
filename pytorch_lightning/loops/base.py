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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from deprecate import void

import pytorch_lightning as pl
from pytorch_lightning.trainer.progress import BaseProgress, Progress
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class Loop(ABC):
    """
    Basic Loops interface. All classes derived from this must implement the following properties and methods:

        * :attr:`done` (property): Condition to break the loop
        * :attr:`reset` (method): Resets the internal state between multiple calls of :attr:`run`
        * :attr:`advance` (method): Implements one step of the loop

    This class implements the following loop structure:

    .. codeblock:: python

        on_run_start()

        while not done:
            on_advance_start()
            advance()
            on_advance_end()

        on_run_end()
    """

    def __init__(self) -> None:
        # TODO: replace by progress tracking
        self.iteration_count: int = 0
        self.restarting = False
        self._trainer: Optional["pl.Trainer"] = None

    @property
    def trainer(self) -> Optional["pl.Trainer"]:
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: "pl.Trainer"):
        """Connects this loop's trainer and its children"""
        if not isinstance(trainer, pl.Trainer):
            raise MisconfigurationException(
                f"Loop {self.__class__.__name__} should be connected to a `Trainer`, found: {trainer}."
            )
        self._trainer = trainer
        for v in self.__dict__.values():
            if isinstance(v, Loop):
                v.trainer = trainer

    @property
    @abstractmethod
    def done(self) -> bool:
        """Property indicating when loop is finished"""

    @property
    def skip(self) -> bool:
        """Determine whether to return immediately from the call to :meth:`run`."""
        return False

    def connect(self, **kwargs: "Loop") -> None:
        """Optionally connect one or multiple loops to this one. Linked loops should form a tree."""

    def on_skip(self) -> Optional[Any]:
        """
        The function to run when :meth:`run` should be skipped, determined by the condition in :attr:`skip`.

        Returns:
            the default output value of :meth:`on_run_end`
        """

    def run(self, *args: Any, **kwargs: Any) -> Optional[Any]:
        """
        The main entry point to the loop.

        Will frequently check the :attr:`done` condition and calls :attr:`advance`
        until :attr:`done` evaluates to ``True``.

        Returns:
            the output of :attr:`on_run_end` (often outputs collected from each step of the loop)
        """
        if self.skip:
            return self.on_skip()

        self.reset()

        self.on_run_start(*args, **kwargs)

        while not self.done:
            try:
                self.on_advance_start(*args, **kwargs)
                self.advance(*args, **kwargs)
                self.on_advance_end()
                self.iteration_count += 1
                self.restarting = False
            except StopIteration:
                break

        output = self.on_run_end()
        return output

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state of the loop at the beginning of each call to :attr:`run`."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """
        Hook to be called as the first thing after entering :attr:`run` (except the state reset).

        Accepts all arguments passed to :attr:`run`.
        """
        void(*args, **kwargs)

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """
        Hook to be called each time before :attr:`advance` is called. Accepts all arguments passed to :attr`run`.
        """
        void(*args, **kwargs)

    @abstractmethod
    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Performs a single step. Accepts all arguments passed to :attr:`run`."""

    def on_advance_end(self) -> None:
        """Hook to be called each time after :attr:`advance` is called."""

    def on_run_end(self) -> Any:
        """Hook to be called at the end of the run. Its return argument is returned from :attr:`run`."""

    def teardown(self) -> None:
        """Use to release memory etc."""

    def on_save_checkpoint(self) -> Dict:
        """
        Called when saving a model checkpoint, use to persist loop state.

        Returns:
            The current loop state.
        """
        return {}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        """Called when loading a model checkpoint, use to reload loop state."""

    def state_dict(self, destination: Optional[Dict] = None, prefix: Optional[str] = "") -> Dict:
        """
        The state dict is determined by the state and progress of this loop and all its children.

        Args:
            destination: An existing dictionary to update with this loop's state. By default a new dictionary
                is returned.
            prefix: A prefix for each key in the state dictionary
        """
        if destination is None:
            destination = {}

        destination[prefix + "state_dict"] = self.on_save_checkpoint()

        for k, v in self.__dict__.items():
            if isinstance(v, BaseProgress):
                destination[prefix + k] = v.state_dict()
            elif isinstance(v, Loop):
                v.state_dict(destination, prefix + k + ".")

        return destination

    def load_state_dict(self, state_dict: Dict, prefix: str = "", restart_progress: bool = True) -> None:
        """Loads the state of this loop and all its children."""
        self._load_from_state_dict(state_dict.copy(), prefix, restart_progress)
        for k, v in self.__dict__.items():
            if isinstance(v, Loop):
                v.load_state_dict(state_dict.copy(), prefix + k + ".", restart_progress)

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, restart_progress: bool) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, BaseProgress):
                v.load_state_dict(state_dict[prefix + k])
                if restart_progress:
                    apply_to_collection(v, Progress, lambda p: p.current.reset_on_restart())
        self.on_load_checkpoint(state_dict[prefix + "state_dict"])
        self.restarting = True
