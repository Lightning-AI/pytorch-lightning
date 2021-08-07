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
"""Warning-related utilities"""
import warnings
from functools import partial
from typing import Any, Sequence, Tuple, Type, Union

from pytorch_lightning.utilities.distributed import rank_zero_only


def _warn(m: Union[str, Warning], category: Union[Type[Warning], Any], stacklevel: int = 2, **kwargs: Any) -> None:
    warnings.warn(m, category, stacklevel=stacklevel, **kwargs)


@rank_zero_only
def rank_zero_warn(
    m: Union[str, Warning], category: Union[Type[Warning], Any], stacklevel: int = 4, **kwargs: Any
) -> None:
    _warn(m, category, stacklevel=stacklevel, **kwargs)


class LightningDeprecationWarning(DeprecationWarning):
    ...


# enable our warnings
warnings.simplefilter("default", LightningDeprecationWarning)

rank_zero_deprecation = partial(rank_zero_warn, category=LightningDeprecationWarning)


class WarningCache(set):
    def warn(self, m: str, *args: Any, stacklevel: int = 5, **kwargs: Any) -> None:
        if m not in self:
            self.add(m)
            rank_zero_warn(m, *args, stacklevel=stacklevel, **kwargs)

    def deprecation(self, m: str, *args: Any, stacklevel: int = 5, **kwargs: Any) -> None:
        if m not in self:
            self.add(m)
            rank_zero_deprecation(m, *args, stacklevel=stacklevel, **kwargs)
