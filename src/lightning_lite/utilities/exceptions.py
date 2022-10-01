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

from typing import Any, Type

from lightning_utilities.core.rank_zero import rank_zero_deprecation


class _DeprecatedException(type):
    def __instancecheck__(cls: Type, instance: object) -> bool:
        # https://peps.python.org/pep-3119/
        return any(cls.__subclasscheck__(c, stacklevel=7) for c in {type(instance), instance.__class__})

    def __subclasscheck__(cls, subclass: Type, stacklevel: int = 5) -> bool:
        mro = subclass.mro()
        real = mro[1]
        if cls is not subclass and cls is real:
            raise TypeError(f"`{subclass.__name__}` should not be used for checks.")
        if cls is MisconfigurationException:
            rank_zero_deprecation(
                f"`{MisconfigurationException.__name__}` is deprecated. Please check with `{real.__name__}` instead.",
                stacklevel=stacklevel,
            )
        return cls in mro


class MisconfigurationException(Exception, metaclass=_DeprecatedException):
    """Exception used to inform users of misuse with PyTorch Lightning."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(f"Using `{type(self).__name__}` is deprecated.", stacklevel=5)
        super().__init__(*args, **kwargs)
