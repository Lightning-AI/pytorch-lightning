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
from typing import Any, Optional
from weakref import proxy

from deprecate import void

import pytorch_lightning as pl


class Loop(ABC):
    """
    Basic Loops interface. All classes derived from this must implement the following properties and methods:

        * :attr`done` (property): Condition to break the loop
        * :attr`reset` (method): Resets the internal state between multiple calls of :attr`run`
        * :attr`advance` (method): Implements one step of the loop

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

    @property
    @abstractmethod
    def done(self) -> bool:
        """Property indicating when loop is finished"""

    def connect(self, trainer: 'pl.Trainer', *args: Any, **kwargs: Any) -> None:
        """Connects Loop with all the necessary things like connectors and accelerators."""
        self.trainer = proxy(trainer)

    @abstractmethod
    def reset(self) -> None:
        """Resets the internal state of the loop at the beginning of each call to :attr:`run`."""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        The main entry point to the loop.

        Will frequently check the :attr:`done` condition and calls :attr:`advance`
        until :attr`done` evaluates to ``True``.

        Returns:
            the output of :attr`on_run_end` (often outputs collected from each step of the loop)
        """
        self.reset()
        self.on_run_start(*args, **kwargs)

        while not self.done:
            try:
                self.on_advance_start(*args, **kwargs)
                self.advance(*args, **kwargs)
                self.on_advance_end()
                self.iteration_count = self.increment_iteration(self.iteration_count)
            except StopIteration:
                break

        return self.on_run_end()

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
        """
        Performs a single step. Accepts all arguments passed to :attr:`run`.

        """

    def on_advance_end(self) -> None:
        """Hook to be called each time after :attr:`advance` is called."""

    def on_run_end(self) -> Any:
        """Hook to be called at the end of the run. Its return argument is returned from :attr:`run`.

        Returns:
            The returned value from the whole :attr:`run` (typically some aggregated outputs from the loop steps).
        """

    def increment_iteration(self, iteration: int) -> int:
        """Helper Function to increment the iteration count.
        Can be used to increment other counters at the same time.

        Args:
            iteration: The current iteration (before increment)

        Returns:
            the incremented iteration count.
        """
        return iteration + 1
