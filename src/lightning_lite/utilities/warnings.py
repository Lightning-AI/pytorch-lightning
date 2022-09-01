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
"""Warning-related utilities."""

import warnings
from typing import Any

from lightning_lite.utilities import rank_zero_deprecation, rank_zero_warn
from lightning_lite.utilities.rank_zero import LightningDeprecationWarning, rank_zero_info

# enable our warnings
warnings.simplefilter("default", category=LightningDeprecationWarning)


class PossibleUserWarning(UserWarning):
    """Warnings that could be false positives."""


class WarningCache(set):
    def warn(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        if message not in self:
            self.add(message)
            rank_zero_warn(message, stacklevel=stacklevel, **kwargs)

    def deprecation(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        if message not in self:
            self.add(message)
            rank_zero_deprecation(message, stacklevel=stacklevel, **kwargs)

    def info(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        if message not in self:
            self.add(message)
            rank_zero_info(message, stacklevel=stacklevel, **kwargs)
