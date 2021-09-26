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

import os
from functools import wraps
from typing import Any, Callable, Optional

import pytorch_lightning as pl


def enabled_only(fn: Callable) -> Optional[Callable]:
    """Decorate a logger method to run it only on the process with rank 0.

    Args:
        fn: Function to decorate
    """

    @wraps(fn)
    def wrapped_fn(self: Callable, *args: Any, **kwargs: Any) -> Optional[Any]:
        if self.enabled:
            fn(self, *args, **kwargs)
        return None

    return wrapped_fn


class InternalDebugger:
    def __init__(self, trainer: "pl.Trainer") -> None:
        self.enabled = os.environ.get("PL_DEV_DEBUG", "0") == "1"
        self.trainer = trainer
