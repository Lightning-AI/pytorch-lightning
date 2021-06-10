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

import pytorch_lightning as pl


class Loop(ABC):

    def __init__(self):
        self.iteration_count: int = 0
        self.trainer: Optional['pl.Trainer'] = None

    @property
    @abstractmethod
    def done(self) -> bool:
        """Property indicating when loop is finished"""

    def connect(self, trainer, *args, **kwargs) -> None:
        """Connects Loop with all the necessary things like connectors and accelerators"""
        self.trainer = proxy(trainer)

    @abstractmethod
    def reset(self) -> None:
        pass

    def run(self, *args: Any, **kwargs: Any) -> Any:
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
