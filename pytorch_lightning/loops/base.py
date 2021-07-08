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
from typing import Any, Dict, Optional, OrderedDict

from deprecate import void

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
        self.trainer: Optional['pl.Trainer'] = None
        self._cached_state: Optional[Dict] = None
        self.restarting = False
        self._loops = OrderedDict()
        self._progress = OrderedDict()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Loop):
            self._loops[name] = value
        elif isinstance(value, BaseProgress):
            self._progress[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name) -> Any:
        loops = self.__dict__.get('_loops')
        if loops is None:
            raise MisconfigurationException("The Loop wasn't called parent `__init__` function.")

        if name in loops:
            return loops[name]

        progress = self.__dict__.get('_progress')

        if name in progress:
            return progress[name]

        if name not in self.__dict__:
            raise AttributeError(f"{self.__class__.__name__} Loop doesn't have attribute {name}.")

        return self.__dict__[name]

    def __delattr__(self, name) -> None:
        if name in self._loops:
            del self._loops[name]
        elif name in self._progress:
            del self._progress[name]
        else:
            object.__delattr__(self, name)

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
        if not isinstance(trainer, pl.Trainer):
            raise MisconfigurationException(
                f"Loop {self.__class__.__name__} should be connected to a `Trainer`, found: {trainer}."
            )
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
        if self.skip:
            return self.on_skip()

        if self.restarting:
            self.restore(self._cached_state)
            self._cached_state = None
            self.restarting = False
        else:
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
    def restore(self, state: Optional[Dict] = None) -> None:
        """Restore the internal state of the loop the beginning of run if restarting is ``True``."""

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

    @abstractmethod
    def state_dict(self) -> Dict:
        """Current Loop state"""

    def get_state_dict(self, destination: Optional[OrderedDict] = None, prefix: Optional[str] = '') -> OrderedDict:
        if destination is None:
            destination = OrderedDict()

        destination[prefix + "state_dict"] = self.state_dict()

        for name, progress in self._progress.items():
            destination[prefix + name] = progress.state_dict()

        for name, loop in self._loops.items():
            loop.get_state_dict(destination, prefix + name + '.')
        return destination

    def _load_from_state_dict(self, state_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs):
        self._cached_state = state_dict[prefix + "state_dict"]

        for name, progress in self._progress.items():
            progress.load_state_dict(state_dict[prefix + name])

    def load_state_dict(self, state_dict: Dict, strict: bool = True):

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        state_dict = state_dict.copy()

        def load(loop, prefix=''):
            loop._load_from_state_dict(state_dict, prefix, True, missing_keys, unexpected_keys, error_msgs)
            loop.restarting = True
            for name, loop_children in loop._loops.items():
                if loop_children is not None:
                    load(loop_children, prefix + name + '.')

        load(self)
        load = None  # break load->load reference cycle
