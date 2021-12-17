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
from functools import partial
from typing import Any, Union

from pytorch_lightning.utilities.distributed import rank_zero_only


def _warn(message: Union[str, Warning], stacklevel: int = 2, **kwargs: Any) -> None:
    if type(stacklevel) is type and issubclass(stacklevel, Warning):
        rank_zero_deprecation(
            "Support for passing the warning category positionally is deprecated in v1.6 and will be removed in v1.8"
            f" Please, use `category={stacklevel.__name__}`."
        )
        kwargs["category"] = stacklevel
        stacklevel = kwargs.pop("stacklevel", 2)
    warnings.warn(message, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_warn(message: Union[str, Warning], stacklevel: int = 4, **kwargs: Any) -> None:
    _warn(message, stacklevel=stacklevel, **kwargs)


class PossibleUserWarning(UserWarning):
    """Warnings that could be false positives."""


class LightningDeprecationWarning(DeprecationWarning):
    """Deprecation warnings raised by PyTorch Lightning."""


# enable our warnings
warnings.simplefilter("default", category=LightningDeprecationWarning)

rank_zero_deprecation = partial(rank_zero_warn, category=LightningDeprecationWarning)


class WarningCache(set):
    def warn(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        if message not in self:
            self.add(message)
            rank_zero_warn(message, stacklevel=stacklevel, **kwargs)

    def deprecation(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        if message not in self:
            self.add(message)
            rank_zero_deprecation(message, stacklevel=stacklevel, **kwargs)
