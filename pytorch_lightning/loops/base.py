from _weakref import proxy
from abc import ABCMeta, abstractmethod
from typing import Any, Counter, List, Optional

import pytorch_lightning as pl


class Loop(metaclass=ABCMeta):

    def __init__(self):
        self.iteration_count: int = 0
        self.trainer: Optional['pl.Trainer'] = None

    @abstractmethod
    def connect(self, trainer, *args, **kwargs):
        """Connects Loop with all the necessary things like connectors and accelerators"""
        self.trainer = proxy(trainer)

    @property
    @abstractmethod
    def done(self):
        """Property indicating when loop is finished"""

    @abstractmethod
    def advance(self, *args: Any, **kwargs: Any):
        """What to do within a single step"""

    def on_run_start(self, *args: Any, **kwargs: Any):
        pass

    def on_run_end(self, outputs: List) -> List:
        return outputs

    def on_advance_start(self, *args: Any, **kwargs: Any):
        pass

    def on_advance_end(self, curr_output: Any) -> Any:
        return curr_output

    def run(self, *args: Any, **kwargs: Any):
        self.on_run_start(*args, **kwargs)

        outputs = []

        while not self.done:

            self.on_advance_start(*args, **kwargs)
            curr_output = self.advance(*args, **kwargs)
            curr_output = self.on_advance_end(curr_output)

            outputs.append(curr_output)

            self.iteration_count += 1

        outputs = self.on_run_end(outputs)
        return outputs

    def state_dict(self):
        return dict()
