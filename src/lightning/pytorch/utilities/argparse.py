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
"""Utilities for Argument Parsing within Lightning Components."""

import inspect
import os
from argparse import Namespace
from ast import literal_eval
from contextlib import suppress
from functools import wraps
from typing import Any, Callable, Type, TypeVar, cast

_T = TypeVar("_T", bound=Callable[..., Any])


def _parse_env_variables(cls: Type, template: str = "PL_%(cls_name)s_%(cls_argument)s") -> Namespace:
    """Parse environment arguments if they are defined.

    Examples:

        >>> from lightning.pytorch import Trainer
        >>> _parse_env_variables(Trainer)
        Namespace()
        >>> import os
        >>> os.environ["PL_TRAINER_DEVICES"] = '42'
        >>> os.environ["PL_TRAINER_BLABLABLA"] = '1.23'
        >>> _parse_env_variables(Trainer)
        Namespace(devices=42)
        >>> del os.environ["PL_TRAINER_DEVICES"]

    """
    env_args = {}
    for arg_name in inspect.signature(cls).parameters:
        env = template % {"cls_name": cls.__name__.upper(), "cls_argument": arg_name.upper()}
        val = os.environ.get(env)
        if not (val is None or val == ""):
            # todo: specify the possible exception
            with suppress(Exception):
                # converting to native types like int/float/bool
                val = literal_eval(val)
            env_args[arg_name] = val
    return Namespace(**env_args)


def _defaults_from_env_vars(fn: _T) -> _T:
    @wraps(fn)
    def insert_env_defaults(self: Any, *args: Any, **kwargs: Any) -> Any:
        cls = self.__class__  # get the class
        if args:  # in case any args passed move them to kwargs
            # parse the argument names
            cls_arg_names = inspect.signature(cls).parameters
            # convert args to kwargs
            kwargs.update(dict(zip(cls_arg_names, args)))
        env_variables = vars(_parse_env_variables(cls))
        # update the kwargs by env variables
        kwargs = dict(list(env_variables.items()) + list(kwargs.items()))

        # all args were already moved to kwargs
        return fn(self, **kwargs)

    return cast(_T, insert_env_defaults)
