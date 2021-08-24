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
from typing import Callable, Optional


def is_param_in_hook_signature(
    hook_fx: Callable, param: str, explicit: bool = False, min_args: Optional[int] = None
) -> bool:
    """
    Args:
        hook_fx: the hook callable
        param: the name of the parameter to check
        explicit: whether the parameter has to be explicitly declared
        min_args: whether the `signature` as at least `min_args` parameters
    """
    hook_params = list(inspect.signature(hook_fx).parameters)
    return (
        param in hook_params
        or (not explicit and "args" in hook_params)
        or (isinstance(min_args, int) and len(hook_params) >= min_args)
    )
