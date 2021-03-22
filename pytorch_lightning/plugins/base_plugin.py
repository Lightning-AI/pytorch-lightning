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


class Plugin(ABC):
    """Basic Plugin class to derive precision and training type plugins from."""

    def pre_dispatch(self) -> None:
        """Hook to do something before the training/evaluation/prediction starts."""

    def post_dispatch(self) -> None:
        """Hook to do something after the training/evaluation/prediction finishes."""

    @contextlib.contextmanager
    def forward_context(self) -> Generator:
        """A contextmanager for managing model forward/training_step/evaluation_step/predict_step"""
        yield

