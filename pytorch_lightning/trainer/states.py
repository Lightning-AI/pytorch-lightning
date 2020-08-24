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

from enum import Enum
from functools import wraps
from typing import Callable, Optional

import pytorch_lightning


class TrainerState(Enum):
    """ State which is set in the :class:`~pytorch_lightning.trainer.trainer.Trainer`
    to indicate what is currently or was executed. """
    INITIALIZING = 'INITIALIZING'
    RUNNING = 'RUNNING'
    FINISHED = 'FINISHED'
    INTERRUPTED = 'INTERRUPTED'


def trainer_state(*, entering: Optional[TrainerState] = None, exiting: Optional[TrainerState] = None) -> Callable:
    """ Decorator for :class:`~pytorch_lightning.trainer.trainer.Trainer` methods
    which changes state to `entering` before the function execution and `exiting`
    after the function is executed. If `None` is passed to `entering`, the state is not changed.
    If `None` is passed to `exiting`, the state is restored to the state before function execution.
    If `INTERRUPTED` state is set inside a run function, the state remains `INTERRUPTED`.
    """

    def wrapper(fn) -> Callable:
        @wraps(fn)
        def wrapped_fn(self, *args, **kwargs):
            if not isinstance(self, pytorch_lightning.Trainer):
                return fn(self, *args, **kwargs)

            state_before = self._state
            if entering is not None:
                self._state = entering
            result = fn(self, *args, **kwargs)

            # The INTERRUPTED state can be set inside the run function. To indicate that run was interrupted
            # we retain INTERRUPTED state
            if self._state == TrainerState.INTERRUPTED:
                return result

            self._state = exiting if exiting is not None else state_before
            return result

        return wrapped_fn

    return wrapper
