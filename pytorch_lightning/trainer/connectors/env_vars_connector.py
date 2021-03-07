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

from functools import wraps
from typing import Callable

from pytorch_lightning.utilities.argparse import get_init_arguments_and_types, parse_env_variables


def overwrite_by_env_vars(fn: Callable) -> Callable:
    """
    Decorator for :class:`~pytorch_lightning.trainer.trainer.Trainer` methods for which
    input arguments should be moved automatically to the correct device.

    """

    @wraps(fn)
    def overwrite_by_env_vars(self, *args, **kwargs):
        # get the class
        cls = self.__class__
        if args:  # inace any args passed move them to kwargs
            # parse only the argument names
            cls_arg_names = [arg[0] for arg in get_init_arguments_and_types(cls)]
            # convert args to kwargs
            kwargs.update({k: v for k, v in zip(cls_arg_names, args)})
        # update the kwargs by env variables
        # todo: maybe add a warning that some init args were overwritten by Env arguments
        kwargs.update(vars(parse_env_variables(cls)))

        # all args were already moved to kwargs
        return fn(self, **kwargs)

    return overwrite_by_env_vars
