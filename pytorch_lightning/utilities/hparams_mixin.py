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
import copy
import inspect
import re
from argparse import Namespace
from typing import Union, Any

from pytorch_lightning.core.saving import PRIMITIVE_TYPES, ALLOWED_CONFIG_TYPES
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.parsing import get_init_args


class HyperparametersMixin:

    __jit_unused_properties__ = [
        "hparams",
        "hparams_initial"
    ]

    def save_hyperparameters(self, *args, frame=None) -> None:
        """Save all model arguments.

        Args:
            args: single object of `dict`, `NameSpace` or `OmegaConf`
             or string names or argumenst from class `__init__`

        >>> from collections import OrderedDict
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
        """
        if not frame:
            frame = inspect.currentframe().f_back
        init_args = get_init_args(frame)
        assert init_args, "failed to inspect the self init"
        if not args:
            # take all arguments
            hp = init_args
            self._hparams_name = "kwargs" if hp else None
        else:
            # take only listed arguments in `save_hparams`
            isx_non_str = [i for i, arg in enumerate(args) if not isinstance(arg, str)]
            if len(isx_non_str) == 1:
                hp = args[isx_non_str[0]]
                cand_names = [k for k, v in init_args.items() if v == hp]
                self._hparams_name = cand_names[0] if cand_names else None
            else:
                hp = {arg: init_args[arg] for arg in args if isinstance(arg, str)}
                self._hparams_name = "kwargs"

        # `hparams` are expected here
        if hp:
            self._set_hparams(hp)
        # make deep copy so  there is not other runtime changes reflected
        self._hparams_initial = copy.deepcopy(self._hparams)

    def _set_hparams(self, hp: Union[dict, Namespace, str]) -> None:
        hp = self._to_hparams_dict(hp)

        if isinstance(hp, dict) and isinstance(self.hparams, dict):
            self.hparams.update(hp)
        else:
            self._hparams = hp

    def _to_hparams_dict(self, hp):
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
    def hparams(self) -> Union[AttributeDict, str]:
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @property
    def hparams_initial(self) -> AttributeDict:
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        # prevent any change
        return copy.deepcopy(self._hparams_initial)

    @hparams.setter
    def hparams(self, hp: Union[dict, Namespace, Any]):
        hparams_assignment_name = self.__get_hparams_assignment_variable()
        self._hparams_name = hparams_assignment_name
        self._set_hparams(hp)
        # this resolves case when user does not uses `save_hyperparameters` and do hard assignement in init
        if not hasattr(self, "_hparams_initial"):
            self._hparams_initial = copy.deepcopy(self._hparams)

    def __get_hparams_assignment_variable(self):
        """
        looks at the code of the class to figure out what the user named self.hparams
        this only happens when the user explicitly sets self.hparams
        """
        try:
            class_code = inspect.getsource(self.__class__)
            lines = class_code.split("\n")
            for line in lines:
                line = re.sub(r"\s+", "", line, flags=re.UNICODE)
                if ".hparams=" in line:
                    return line.split("=")[1]
        except Exception as e:
            return "hparams"

        return None
