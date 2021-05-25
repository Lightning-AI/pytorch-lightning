from abc import ABC, abstractmethod
from typing import Any, Optional

from _weakref import proxy

import pytorch_lightning as pl


class Loop(ABC):

    def __init__(self):
        self.iteration_count: int = 0
        self.trainer: Optional['pl.Trainer'] = None

    @abstractmethod
    def connect(self, trainer, *args, **kwargs):
        """Connects Loop with all the necessary things like connectors and accelerators"""
        self.trainer = proxy(trainer)

    @property
    @abstractmethod
    def done(self) -> bool:
        """Property indicating when loop is finished"""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.on_run_start(*args, **kwargs)

        while not self.done:

            self.on_advance_start(*args, **kwargs)
            self.advance(*args, **kwargs)
            self.on_advance_end()
            self.iteration_count += 1

        return self.on_run_end()

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def advance(self, *args: Any, **kwargs: Any) -> None:
        """What to do within a single step"""

    def on_advance_end(self) -> None:
        pass

    def on_run_end(self) -> Any:
        pass

    def state_dict(self):
        return dict()
