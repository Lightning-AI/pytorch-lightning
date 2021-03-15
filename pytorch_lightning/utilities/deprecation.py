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
import inspect
from functools import wraps
from typing import Any, Callable, List, Tuple

from pytorch_lightning.utilities import rank_zero_warn


def get_func_arguments_and_types(func: Callable) -> List[Tuple[str, Tuple, Any]]:
    """Parse function arguments, types and default values

    Example:
        >>> get_func_arguments_and_types(get_func_arguments_and_types)
        [('func', typing.Callable, <class 'inspect._empty'>)]
    """
    func_default_params = inspect.signature(func).parameters
    name_type_default = []
    for arg in func_default_params:
        arg_type = func_default_params[arg].annotation
        arg_default = func_default_params[arg].default
        name_type_default.append((arg, arg_type, arg_default))
    return name_type_default


def deprecated_func(target_func: Callable, ver_deprecate: str = "", ver_remove: str = "") -> Callable:

    def inner_function(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            target_func_str = f'{target_func.__module__}.{target_func.__name__}'
            rank_zero_warn(
                f"This `{func.__name__}` was deprecated since v{ver_deprecate} in favor of `{target_func_str}`."
                f" It will be removed in v{ver_remove}.", DeprecationWarning
            )
            if args:  # in case any args passed move them to kwargs
                # parse only the argument names
                cls_arg_names = [arg[0] for arg in get_func_arguments_and_types(func)]
                # convert args to kwargs
                kwargs.update({k: v for k, v in zip(cls_arg_names, args)})
            target_args = [arg[0] for arg in get_func_arguments_and_types(target_func)]
            assert all(arg in target_args for arg in kwargs), \
                "Failed mapping, arguments missing in target func: %s" % [arg not in target_args for arg in kwargs]
            # all args were already moved to kwargs
            return target_func(**kwargs)

        return wrapper

    return inner_function
