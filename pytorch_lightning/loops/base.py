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
from torch.nn.modules.module import _IncompatibleKeys

import pytorch_lightning as pl
from pytorch_lightning.trainer.progress import BaseProgress
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
        self.iteration_count: int = 0
        self._trainer: Optional['pl.Trainer'] = None
        self.restarting = False

    @property
    def loop_progress(self) -> Dict[str, Any]:
        """Return the progress for the current loop and children loop."""
        progress = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BaseProgress):
                progress[k] = v
            elif isinstance(v, Loop):
                progress[k] = v.loop_progress
        return progress

    @property
    def trainer(self) -> Optional['pl.Trainer']:
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: 'pl.Trainer'):
        """Connect the Trainer to itself and all sub-children loops"""
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

    def connect(self, trainer: 'pl.Trainer', *args: Any, **kwargs: Any) -> None:
        """Connects Loop with all the necessary things like connectors and accelerators."""
        # TODO(@justusschock): Make the trainer a weakref/proxy
        self.trainer = trainer

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
        if self.trainer is None:
            raise MisconfigurationException(f"The {self.__class__.__name__} Loop hasn't been attached to any Trainer.")

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

    def on_load_checkpoint(self, state_dict: Dict):
        """Called when loading a model checkpoint, use to reload loop state."""

    def state_dict(self, destination: Optional[Dict] = None, prefix: Optional[str] = '') -> Dict:
        if destination is None:
            destination = {}

        destination[prefix + "state_dict"] = self.on_save_checkpoint()

        for k, v in self.__dict__.items():
            if isinstance(v, BaseProgress):
                destination[prefix + k] = v.state_dict()
            elif isinstance(v, Loop):
                v.state_dict(destination, prefix + k + '.')

        return destination

    def _load_from_state_dict(
        self, state_dict, prefix, strict, restart_progress, missing_keys, unexpected_keys, error_msgs
    ):
        for k, v in self.__dict__.items():
            if isinstance(v, BaseProgress):
                v.load_state_dict(state_dict[prefix + k])

        self.on_load_checkpoint(state_dict[prefix + "state_dict"])

    def load_state_dict(self, state_dict: Dict, restart_progress: bool = True, strict: bool = True):
        """
        This function is highly inspired from ``PyTorch nn.Module``.
        """

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        state_dict = state_dict.copy()

        def load(loop, prefix=''):
            if loop.restarting:
                return
            loop._load_from_state_dict(
                state_dict, prefix, True, restart_progress, missing_keys, unexpected_keys, error_msgs
            )
            loop.restarting = True
            for k, v in self.__dict__.items():
                if isinstance(v, Loop):
                    load(v, prefix + k + '.')

        load(self)

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)
                    )
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys))
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, "\n\t".join(error_msgs))
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)
