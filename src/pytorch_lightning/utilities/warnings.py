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

from pytorch_lightning.utilities.rank_zero import LightningDeprecationWarning as NewLightningDeprecationWarning
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation as new_rank_zero_deprecation
from pytorch_lightning.utilities.rank_zero import rank_zero_warn as new_rank_zero_warn

# enable our warnings
warnings.simplefilter("default", category=NewLightningDeprecationWarning)


class PossibleUserWarning(UserWarning):
    """Warnings that could be false positives."""


class WarningCache(set):
    def warn(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        if message not in self:
            self.add(message)
            new_rank_zero_warn(message, stacklevel=stacklevel, **kwargs)

    def deprecation(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        if message not in self:
            self.add(message)
            new_rank_zero_deprecation(message, stacklevel=stacklevel, **kwargs)


def rank_zero_warn(*args: Any, **kwargs: Any) -> Any:
    new_rank_zero_deprecation(
        "pytorch_lightning.utilities.warnings.rank_zero_warn has been deprecated in v1.6"
        " and will be removed in v1.8."
        " Use the equivalent function from the pytorch_lightning.utilities.rank_zero module instead."
    )
    return new_rank_zero_warn(*args, **kwargs)


def rank_zero_deprecation(*args: Any, **kwargs: Any) -> Any:
    new_rank_zero_deprecation(
        "pytorch_lightning.utilities.warnings.rank_zero_deprecation has been deprecated in v1.6"
        " and will be removed in v1.8."
        " Use the equivalent function from the pytorch_lightning.utilities.rank_zero module instead."
    )
    return new_rank_zero_deprecation(*args, **kwargs)


class LightningDeprecationWarning(NewLightningDeprecationWarning):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        new_rank_zero_deprecation(
            "pytorch_lightning.utilities.warnings.LightningDeprecationWarning has been deprecated in v1.6"
            " and will be removed in v1.8."
            " Use the equivalent class from the pytorch_lightning.utilities.rank_zero module instead."
        )
        super().__init__(*args, **kwargs)
