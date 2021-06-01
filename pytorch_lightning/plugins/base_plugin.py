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
from abc import ABC
from typing import Generator

import pytorch_lightning as pl


class Plugin(ABC):
    """Basic class for all precision- and training type plugins."""

    def pre_dispatch(self) -> None:
        """Hook to do something before the training/evaluation/prediction starts."""

    def dispatch(self, trainer: "pl.Trainer") -> None:
        """Hook to do something at trainer run_stage starts."""

    def post_dispatch(self) -> None:
        """Hook to do something after the training/evaluation/prediction finishes."""

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

    @contextlib.contextmanager
    def predict_step_context(self) -> Generator:
        """A contextmanager for the predict step"""
        yield
