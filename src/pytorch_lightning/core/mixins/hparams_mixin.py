# Copyright The Lightning AI team.
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
import copy
import inspect
import types
from argparse import Namespace
from typing import Any, List, MutableMapping, Optional, Sequence, Union

from pytorch_lightning.core.saving import ALLOWED_CONFIG_TYPES, PRIMITIVE_TYPES
from pytorch_lightning.utilities.parsing import AttributeDict, save_hyperparameters


class HyperparametersMixin:

    __jit_unused_properties__: List[str] = ["hparams", "hparams_initial"]

    def __init__(self) -> None:
        super().__init__()
        self._log_hyperparams = False

    def save_hyperparameters(
        self,
        *args: Any,
        ignore: Optional[Union[Sequence[str], str]] = None,
        frame: Optional[types.FrameType] = None,
        logger: bool = True,
    ) -> None:
        """Save arguments to ``hparams`` attribute.

        Args:
            args: single object of `dict`, `NameSpace` or `OmegaConf`
                or string names or arguments from class ``__init__``
            ignore: an argument name or a list of argument names from
                class ``__init__`` to be ignored
            frame: a frame object. Default is None
            logger: Whether to send the hyperparameters to the logger. Default: True

        Example::
            >>> from pytorch_lightning.core.mixins import HyperparametersMixin
            >>> class ManuallyArgsModel(HyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # manually assign arguments
            ...         self.save_hyperparameters('arg1', 'arg3')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14

            >>> from pytorch_lightning.core.mixins import HyperparametersMixin
            >>> class AutomaticArgsModel(HyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # equivalent automatic
            ...         self.save_hyperparameters()
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = AutomaticArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg2": abc
            "arg3": 3.14

            >>> from pytorch_lightning.core.mixins import HyperparametersMixin
            >>> class SingleArgModel(HyperparametersMixin):
            ...     def __init__(self, params):
            ...         super().__init__()
            ...         # manually assign single argument
            ...         self.save_hyperparameters(params)
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = SingleArgModel(Namespace(p1=1, p2='abc', p3=3.14))
            >>> model.hparams
            "p1": 1
            "p2": abc
            "p3": 3.14

            >>> from pytorch_lightning.core.mixins import HyperparametersMixin
            >>> class ManuallyArgsModel(HyperparametersMixin):
            ...     def __init__(self, arg1, arg2, arg3):
            ...         super().__init__()
            ...         # pass argument(s) to ignore as a string or in a list
            ...         self.save_hyperparameters(ignore='arg2')
            ...     def forward(self, *args, **kwargs):
            ...         ...
            >>> model = ManuallyArgsModel(1, 'abc', 3.14)
            >>> model.hparams
            "arg1": 1
            "arg3": 3.14
        """
        self._log_hyperparams = logger
        # the frame needs to be created in this file.
        if not frame:
            current_frame = inspect.currentframe()
            if current_frame:
                frame = current_frame.f_back
        save_hyperparameters(self, *args, ignore=ignore, frame=frame)

    def _set_hparams(self, hp: Union[MutableMapping, Namespace, str]) -> None:
        hp = self._to_hparams_dict(hp)

        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp

    @staticmethod
    def _to_hparams_dict(hp: Union[MutableMapping, Namespace, str]) -> Union[MutableMapping, AttributeDict]:
        if isinstance(hp, Namespace):
            hp = vars(hp)
        if isinstance(hp, dict):
            hp = AttributeDict(hp)
        elif isinstance(hp, PRIMITIVE_TYPES):
            raise ValueError(f"Primitives {PRIMITIVE_TYPES} are not allowed.")
        elif not isinstance(hp, ALLOWED_CONFIG_TYPES):
            raise ValueError(f"Unsupported config type of {type(hp)}.")
        return hp

    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        """The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user.
        For the frozen set of initial hyperparameters, use :attr:`hparams_initial`.

        Returns:
            Mutable hyperparameters dictionary
        """
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @property
    def hparams_initial(self) -> AttributeDict:
        """The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only.
        Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`.

        Returns:
            AttributeDict: immutable initial hyperparameters
        """
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        # prevent any change
        return copy.deepcopy(self._hparams_initial)
