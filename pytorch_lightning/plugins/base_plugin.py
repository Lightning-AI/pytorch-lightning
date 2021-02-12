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
import contextlib
from abc import ABC, abstractmethod
from typing import Any, Generator, Optional, overload, Sequence, Tuple

import torch


class Plugin(ABC):
    """Basic Plugin class to derive precision and training type plugins from."""

    @abstractmethod
    def connect(self, model: torch.nn.Module, *args: Sequence,
                **kwargs: Sequence) -> Optional[Tuple[torch.nn.Module, Sequence, Sequence]]:
        """Connects the plugin with the accelerator (and thereby with trainer and model).
        Will be called by the accelerator.
        """

    def pre_optimizer_step(self, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:
        """Hook to do something before each optimizer step."""

    def post_optimizer_step(self, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:
        """Hook to do something after each optimizer step."""

    def pre_training(self) -> None:
        """Hook to do something before the training starts."""

    def post_training(self) -> None:
        """Hook to do something after the training finishes."""

    @contextlib.contextmanager
    def train_step_context(self) -> Generator:
        """A contextmanager for the trainstep"""
        yield

    @contextlib.contextmanager
    def val_step_context(self) -> Generator:
        """A contextmanager for the validation step"""
        yield

    @contextlib.contextmanager
    def test_step_context(self) -> Generator:
        """A contextmanager for the teststep"""
        yield
