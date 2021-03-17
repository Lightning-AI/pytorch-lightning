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
from typing import Any, Callable, List, Optional, Tuple

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


def deprecated(target: Callable, ver_deprecate: Optional[str] = "", ver_remove: Optional[str] = "") -> Callable:
    """
    Decorate a function or class ``__init__`` with warning message
     and pass all arguments directly to the target class/method.
    """

    def inner_function(base):

        @wraps(base)
        def wrapped_fn(*args, **kwargs):
            is_class = inspect.isclass(target)
            target_func = target.__init__ if is_class else target
            # warn user only once in lifetime
            if not getattr(wrapped_fn, 'warned', False):
                target_str = f'{target.__module__}.{target.__name__}'
                base_name = base.__qualname__.split('.')[-2] if is_class else base.__name__
                base_str = f'{base.__module__}.{base_name}'
                rank_zero_warn(
                    f"`{base_str}` was deprecated since v{ver_deprecate} in favor of `{target_str}`."
                    f" It will be removed in v{ver_remove}.", DeprecationWarning
                )
                wrapped_fn.warned = True

            if args:  # in case any args passed move them to kwargs
                # parse only the argument names
                arg_names = [arg[0] for arg in get_func_arguments_and_types(base)]
                # convert args to kwargs
                kwargs.update({k: v for k, v in zip(arg_names, args)})
            # fill by base defaults
            base_defaults = {arg[0]: arg[2] for arg in get_func_arguments_and_types(base) if arg[2] != inspect._empty}
            kwargs = dict(list(base_defaults.items()) + list(kwargs.items()))

            target_args = [arg[0] for arg in get_func_arguments_and_types(target_func)]
            assert all(arg in target_args for arg in kwargs), \
                "Failed mapping, arguments missing in target base: %s" % [arg not in target_args for arg in kwargs]
            # all args were already moved to kwargs
            return target_func(**kwargs)

        return wrapped_fn

    return inner_function
